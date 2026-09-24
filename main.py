#!/usr/bin/env python3
"""
PDF Interleave + Rotate Left + 2-Up A4 Composer — Web Edition
=============================================================

Single-file FastAPI application that exposes the PDF pipeline
(interleave A1,B1,A2,B2… → rotate 90° counter-clockwise → 2-up A4-landscape)
as a browser-based upload/download tool with a dark Tailwind UI.

Architecture
------------
  • The core PDF transformation is a pure function on bytes, fully
    decoupled from the HTTP layer, so it stays unit-testable and could be
    reused by a CLI or worker without modification.
  • The HTTP layer (FastAPI) handles multipart upload, size caps, path
    sanitisation, and response streaming.
  • The presentation layer is one inline HTML document using Tailwind CSS
    (CDN) with a dark-first palette. No build step, no Node toolchain.

Pipeline invariants
-------------------
  • ``copy.copy`` precedes every ``rotate`` because ``PageObject.rotate``
    mutates the page in place; cloning first keeps A- and B-copies
    independent.
  • All file bytes live in memory (``BytesIO``); nothing touches disk.
  • Uploaded filenames are reduced to their basename (defeating ``../``
    traversal) before use; the server never trusts them for paths.
  • Uploads are read in capped chunks so an oversized file is rejected
    during streaming rather than after it is fully in memory.

Compatibility
-------------
  Python ≥ 3.10. Verified against FastAPI ≥ 0.115, Starlette ≥ 0.47,
  Uvicorn ≥ 0.30, pypdf ≥ 4.0, python-multipart ≥ 0.0.9.

  HTTP status constants are resolved via ``getattr`` with integer
  fallbacks so the same file runs cleanly on both the pre- and post-RFC
  9110 Starlette releases without emitting deprecation warnings.
"""

from __future__ import annotations

import copy
import io
import os
import socket
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Final, Sequence

import starlette.status as _http_status
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pypdf import PageObject, PdfReader, PdfWriter, Transformation
from starlette.concurrency import run_in_threadpool


# ═════════════════════════════════ Constants ═════════════════════════════

# ── A4 geometry (PDF points; 1 pt = 1/72 inch) ───────────────────────────
A4_SHORT_SIDE_PT: Final[float] = 595.276
A4_LONG_SIDE_PT: Final[float] = 841.890
A4_LANDSCAPE_WIDTH_PT: Final[float] = A4_LONG_SIDE_PT
A4_LANDSCAPE_HEIGHT_PT: Final[float] = A4_SHORT_SIDE_PT

# ── Layout ───────────────────────────────────────────────────────────────
PAGES_PER_SHEET: Final[int] = 2
ROTATION_DEGREES_LEFT: Final[int] = -90

# ── Upload limits ────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES: Final[int] = 100 * 1024 * 1024        # 100 MiB per file
MAX_FILES_PER_REQUEST: Final[int] = 20
READ_CHUNK_BYTES: Final[int] = 1 << 20                  # 1 MiB streaming read
ALLOWED_CONTENT_TYPES: Final[frozenset[str]] = frozenset(
    {"application/pdf", "application/x-pdf", "application/octet-stream"}
)
PDF_EXTENSION: Final[str] = ".pdf"

# ── Output naming ────────────────────────────────────────────────────────
TIME_FORMAT: Final[str] = "%H%M%S"
DATE_FORMAT: Final[str] = "%Y%m%d"
FINAL_FILENAME_TEMPLATE: Final[str] = "{stem}_print_{time}_{date}{ext}"
ZIP_FILENAME_TEMPLATE: Final[str] = "pdf_composer_{stamp}.zip"

# ── HTTP status codes ────────────────────────────────────────────────────
# RFC 9110 renamed 413 and 422; Starlette exposes the new names in
# ``__all__`` and keeps the old ones behind a deprecation shim.  Resolving
# only the new names avoids the shim entirely, and the integer fallback
# keeps this file working on pre-RFC Starlette without an ImportError.
HTTP_413_TOO_LARGE: Final[int] = getattr(
    _http_status, "HTTP_413_CONTENT_TOO_LARGE", 413
)
HTTP_415_UNSUPPORTED: Final[int] = getattr(
    _http_status, "HTTP_415_UNSUPPORTED_MEDIA_TYPE", 415
)
HTTP_422_UNPROCESSABLE: Final[int] = getattr(
    _http_status, "HTTP_422_UNPROCESSABLE_CONTENT", 422
)

# ── Server defaults ──────────────────────────────────────────────────────
DEFAULT_HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT: Final[int] = 8000
MAX_PORT: Final[int] = 65535
PORT_SCAN_ATTEMPTS: Final[int] = 20
LOG_LEVEL: Final[str] = "info"

# ── Error messages ───────────────────────────────────────────────────────
ERR_NO_FILES: Final[str] = "No files were uploaded."
ERR_TOO_MANY_FILES: Final[str] = (
    f"Too many files in one request (max {MAX_FILES_PER_REQUEST})."
)
ERR_BAD_CONTENT_TYPE: Final[str] = (
    "Unsupported content type '{ct}'. Only PDF uploads are accepted."
)
ERR_BAD_EXTENSION: Final[str] = "File '{name}' does not have a .pdf extension."
ERR_FILE_TOO_LARGE: Final[str] = (
    "File '{name}' exceeds the {limit_mb} MB upload limit."
)
ERR_EMPTY_UPLOAD: Final[str] = "File '{name}' is empty."
ERR_NOT_A_PDF: Final[str] = "File '{name}' is not a valid PDF."
ERR_ZERO_PAGES: Final[str] = "File '{name}' has zero pages."
ERR_MISSING_FILENAME: Final[str] = "Uploaded file is missing a filename."
ERR_NO_FREE_PORT: Final[str] = (
    "No free port found in range {start}-{end} on {host}."
)
ERR_INVALID_PORT: Final[str] = "Invalid PORT value: {value!r}"
ERR_PORT_RANGE: Final[str] = "PORT out of range: {port}"


# ═════════════════════════════════ Types ═════════════════════════════════

@dataclass(frozen=True)
class ProcessedPdf:
    """Result of processing one uploaded PDF."""

    output_name: str
    content: bytes
    input_pages: int
    output_sheets: int


@dataclass(frozen=True)
class BatchResult:
    """Aggregate result for one HTTP request."""

    files: tuple[ProcessedPdf, ...]
    failures: tuple[str, ...]


# ═════════════════════════════════ PDF pipeline ══════════════════════════

def _interleave_and_rotate(
    source_pages: Sequence[PageObject],
) -> list[PageObject]:
    """
    Return ``[A1, B1, A2, B2, …]`` where each ``Xi`` is an independent
    copy of the source page, rotated 90° counter-clockwise.

    ``copy.copy`` must precede ``rotate`` because ``rotate`` mutates the
    page in place; cloning first keeps the A- and B-copies independent.
    """
    interleaved: list[PageObject] = []
    for page in source_pages:
        for _ in range(PAGES_PER_SHEET):
            clone = copy.copy(page)
            interleaved.append(clone.rotate(ROTATION_DEGREES_LEFT))
    return interleaved


def _fit_scale(
    src_w: float, src_h: float, slot_w: float, slot_h: float
) -> float:
    """Uniform, aspect-preserving scale that fits the source into the slot."""
    if src_w <= 0.0 or src_h <= 0.0:
        return 1.0
    return min(slot_w / src_w, slot_h / src_h)


def _place_page(
    sheet: PageObject,
    page: PageObject,
    slot_w: float,
    slot_h: float,
    x0: float,
    y0: float,
) -> None:
    """Scale and centre ``page`` inside the slot whose bottom-left is (x0, y0)."""
    src_w = float(page.mediabox.width)
    src_h = float(page.mediabox.height)

    scale = _fit_scale(src_w, src_h, slot_w, slot_h)
    new_w = src_w * scale
    new_h = src_h * scale

    translate_x = x0 + (slot_w - new_w) / 2
    translate_y = y0 + (slot_h - new_h) / 2

    sheet.merge_transformed_page(
        page,
        Transformation().scale(scale).translate(translate_x, translate_y),
    )


def _compose_2up(pages: Sequence[PageObject]) -> tuple[PdfWriter, int]:
    """Compose interleaved pages into 2-up A4-landscape sheets."""
    sheet_w = A4_LANDSCAPE_WIDTH_PT
    sheet_h = A4_LANDSCAPE_HEIGHT_PT
    slot_w = sheet_w / PAGES_PER_SHEET
    slot_h = sheet_h
    left_x = 0.0
    right_x = slot_w
    bottom_y = 0.0

    writer = PdfWriter()
    n_pages = len(pages)
    n_sheets = 0

    for start in range(0, n_pages, PAGES_PER_SHEET):
        sheet = PageObject.create_blank_page(width=sheet_w, height=sheet_h)
        _place_page(sheet, pages[start], slot_w, slot_h, left_x, bottom_y)
        if start + 1 < n_pages:
            _place_page(
                sheet, pages[start + 1], slot_w, slot_h, right_x, bottom_y
            )
        writer.add_page(sheet)
        n_sheets += 1

    return writer, n_sheets


def process_pdf_bytes(pdf_bytes: bytes, stem: str) -> ProcessedPdf:
    """
    Apply the full pipeline to ``pdf_bytes`` and return the composed PDF.

    Raises ``ValueError`` with a user-safe message on any recoverable
    failure (corrupt input, zero pages, etc.).
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:  # pypdf raises various low-level errors here
        raise ValueError(ERR_NOT_A_PDF.format(name=stem)) from exc

    if len(reader.pages) == 0:
        raise ValueError(ERR_ZERO_PAGES.format(name=stem))

    interleaved = _interleave_and_rotate(reader.pages)
    writer, n_sheets = _compose_2up(interleaved)

    buffer = io.BytesIO()
    writer.write(buffer)

    now = datetime.now()
    output_name = FINAL_FILENAME_TEMPLATE.format(
        stem=stem,
        time=now.strftime(TIME_FORMAT),
        date=now.strftime(DATE_FORMAT),
        ext=PDF_EXTENSION,
    )

    return ProcessedPdf(
        output_name=output_name,
        content=buffer.getvalue(),
        input_pages=len(reader.pages),
        output_sheets=n_sheets,
    )


# ═════════════════════════════════ Upload validation ══════════════════════

def _safe_stem(filename: str | None) -> str:
    """
    Derive a filesystem-safe stem from a client-supplied filename.

    Strips any directory component (defeating ``../`` traversal), removes
    the extension, and rejects empty results.
    """
    if not filename:
        raise ValueError(ERR_MISSING_FILENAME)
    # Normalise backslashes first so Windows-style paths are also collapsed.
    normalised = filename.replace("\\", "/")
    base = PurePosixPath(normalised).name
    stem = (
        base[: -len(PDF_EXTENSION)]
        if base.lower().endswith(PDF_EXTENSION)
        else base
    )
    return stem.strip().strip(".") or "document"


def _validate_upload_metadata(file: UploadFile, size: int) -> None:
    """Reject uploads whose declared type/extension/size is unacceptable."""
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=HTTP_415_UNSUPPORTED,
            detail=ERR_BAD_CONTENT_TYPE.format(ct=file.content_type),
        )
    name = file.filename or "<unnamed>"
    if not name.lower().endswith(PDF_EXTENSION):
        raise HTTPException(
            status_code=HTTP_415_UNSUPPORTED,
            detail=ERR_BAD_EXTENSION.format(name=name),
        )
    if size <= 0:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE,
            detail=ERR_EMPTY_UPLOAD.format(name=name),
        )
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=HTTP_413_TOO_LARGE,
            detail=ERR_FILE_TOO_LARGE.format(
                name=name,
                limit_mb=MAX_UPLOAD_BYTES // (1024 * 1024),
            ),
        )


async def _read_upload_capped(file: UploadFile) -> bytes:
    """
    Stream ``file`` into memory, aborting as soon as the byte count exceeds
    ``MAX_UPLOAD_BYTES``.  This enforces the cap during the read instead of
    after the whole payload has been buffered.
    """
    chunks: list[bytes] = []
    total = 0
    name = file.filename or "<unnamed>"
    while True:
        chunk = await file.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=HTTP_413_TOO_LARGE,
                detail=ERR_FILE_TOO_LARGE.format(
                    name=name,
                    limit_mb=MAX_UPLOAD_BYTES // (1024 * 1024),
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)


# ═════════════════════════════════ Batch orchestration ═══════════════════

def _process_one(file: UploadFile, data: bytes) -> ProcessedPdf:
    """Run the pipeline for a single validated upload."""
    stem = _safe_stem(file.filename)
    return process_pdf_bytes(data, stem)


def process_uploads(
    files: Sequence[UploadFile], payloads: Sequence[bytes]
) -> BatchResult:
    """
    Process every upload, collecting per-file failures instead of aborting
    the whole request on the first bad input.
    """
    processed: list[ProcessedPdf] = []
    failures: list[str] = []

    for file, data in zip(files, payloads):
        try:
            processed.append(_process_one(file, data))
        except ValueError as exc:
            failures.append(f"{file.filename or '<unnamed>'}: {exc}")

    return BatchResult(files=tuple(processed), failures=tuple(failures))


def _build_response(batch: BatchResult) -> Response:
    """Return a single PDF, a ZIP of many, or a 422 when everything failed."""
    if not batch.files:
        detail = "; ".join(batch.failures) or ERR_NO_FILES
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE, detail=detail
        )

    if len(batch.files) == 1:
        only = batch.files[0]
        return Response(
            content=only.content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{only.output_name}"'
                ),
                "X-Input-Pages": str(only.input_pages),
                "X-Output-Sheets": str(only.output_sheets),
            },
        )

    stamp = datetime.now().strftime(f"{DATE_FORMAT}_{TIME_FORMAT}")
    zip_name = ZIP_FILENAME_TEMPLATE.format(stamp=stamp)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in batch.files:
            archive.writestr(item.output_name, item.content)

    headers = {
        "Content-Disposition": f'attachment; filename="{zip_name}"',
        "X-File-Count": str(len(batch.files)),
    }
    if batch.failures:
        # Surface partial-failure context without breaking the download.
        headers["X-Failed-Count"] = str(len(batch.failures))

    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers=headers,
    )


# ═════════════════════════════════ FastAPI app ═══════════════════════════

app = FastAPI(
    title="PDF Interleave + Rotate + 2-Up Composer",
    description=(
        "Interleave a PDF (A1,B1,A2,B2…), rotate every page 90° "
        "counter-clockwise, and lay the result out as 2-up A4-landscape "
        "sheets."
    ),
    docs_url=None,
    redoc_url=None,
)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single-file UI."""
    return HTMLResponse(content=INDEX_HTML)


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Lightweight readiness probe."""
    return {"status": "ok"}


@app.post("/api/process")
async def api_process(
    files: Annotated[
        list[UploadFile], File(description="One or more PDF files")
    ],
) -> Response:
    """
    Accept one or more PDF uploads, run the pipeline, and return either a
    single PDF or a ZIP archive containing every result.
    """
    if not files:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE, detail=ERR_NO_FILES
        )
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=HTTP_413_TOO_LARGE, detail=ERR_TOO_MANY_FILES
        )

    payloads: list[bytes] = []
    for file in files:
        data = await _read_upload_capped(file)
        _validate_upload_metadata(file, len(data))
        payloads.append(data)

    # pypdf is CPU-bound and synchronous; offload so the event loop stays
    # free to serve other requests.
    batch = await run_in_threadpool(process_uploads, files, payloads)
    return _build_response(batch)


# ═════════════════════════════════ Presentation ══════════════════════════

INDEX_HTML: Final[str] = r"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>PDF Composer — Interleave · Rotate · 2-Up A4</title>
<script src="https://cdn.tailwindcss.com"></script>
<script>
  tailwind.config = {
    darkMode: 'class',
    theme: {
      extend: {
        colors: {
          ink:    { 900:'#0b0f14', 800:'#111820', 700:'#182230', 600:'#1f2b3a' },
          steel:  { 400:'#8aa0b4', 300:'#b6c6d4', 200:'#d7e1ea' },
          signal: { 500:'#22d3ee', 400:'#38dcf2', 600:'#0ea5b7' },
        },
        fontFamily: {
          sans: ['Inter','system-ui','-apple-system','Segoe UI','Roboto','sans-serif'],
          mono: ['JetBrains Mono','ui-monospace','SFMono-Regular','Menlo','monospace'],
        },
      },
    },
  };
</script>
<style>
  :root { color-scheme: dark; }
  body { background:
    radial-gradient(1200px 600px at 15% -10%, rgba(34,211,238,.10), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(56,220,242,.06), transparent 55%),
    #0b0f14; }
  .card { background: rgba(17,24,32,.72); backdrop-filter: blur(10px); }
  .drop-active { border-color:#22d3ee !important; background: rgba(34,211,238,.08); }
  .shimmer { background: linear-gradient(90deg,#1f2b3a 0%,#2a3a4d 50%,#1f2b3a 100%);
    background-size: 200% 100%; animation: sh 1.2s linear infinite; }
  @keyframes sh { to { background-position: -200% 0; } }
  ::-webkit-scrollbar{width:10px;height:10px}
  ::-webkit-scrollbar-thumb{background:#1f2b3a;border-radius:8px}
</style>
</head>
<body class="min-h-screen font-sans text-steel-200 antialiased">

<div class="max-w-5xl mx-auto px-5 py-10 sm:py-14">

  <header class="mb-10">
    <div class="flex items-center gap-3 mb-3">
      <div class="w-10 h-10 rounded-xl bg-signal-500/15 border border-signal-500/30 grid place-items-center">
        <svg class="w-5 h-5 text-signal-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
          <path d="M4 7h6l2 2h8v8a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2z"/>
        </svg>
      </div>
      <h1 class="text-2xl sm:text-3xl font-semibold tracking-tight text-white">PDF Composer</h1>
    </div>
    <p class="text-sm text-steel-400 max-w-2xl leading-relaxed">
      Interleave <span class="font-mono text-steel-300">A1,B1,A2,B2…</span> ·
      Rotate <span class="font-mono text-steel-300">90° left</span> ·
      Compose <span class="font-mono text-steel-300">2-up A4 landscape</span>.
      Everything runs in-memory; nothing is written to the server's disk.
    </p>
  </header>

  <section class="card rounded-2xl border border-ink-600 shadow-2xl shadow-black/40 p-6 sm:p-8 mb-6">
    <div id="dropzone"
         class="rounded-xl border-2 border-dashed border-ink-600 hover:border-signal-500/60 transition-colors cursor-pointer
                px-6 py-12 text-center select-none">
      <input id="fileInput" type="file" accept="application/pdf,.pdf" multiple class="hidden" />
      <div class="mx-auto w-14 h-14 rounded-full bg-ink-700 grid place-items-center mb-4">
        <svg class="w-6 h-6 text-signal-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
          <path d="M12 16V4m0 0L7 9m5-5 5 5M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>
        </svg>
      </div>
      <p class="text-base font-medium text-white mb-1">Drop PDFs here or click to browse</p>
      <p class="text-xs text-steel-400">Up to 20 files · 100 MB each · <span class="font-mono">.pdf</span> only</p>
    </div>

    <div id="fileList" class="mt-5 space-y-2 hidden"></div>

    <div class="mt-6 flex flex-col sm:flex-row gap-3">
      <button id="processBtn" disabled
        class="flex-1 inline-flex items-center justify-center gap-2 rounded-xl px-5 py-3 font-medium
               bg-signal-500 text-ink-900 hover:bg-signal-400 disabled:opacity-40 disabled:cursor-not-allowed
               transition-colors">
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M13 2 3 14h7l-1 8 10-12h-7l1-8z"/>
        </svg>
        Compose PDFs
      </button>
      <button id="clearBtn" disabled
        class="rounded-xl px-5 py-3 font-medium border border-ink-600 text-steel-300
               hover:bg-ink-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
        Clear
      </button>
    </div>
  </section>

  <section id="statusPanel" class="hidden card rounded-2xl border border-ink-600 p-6 mb-6">
    <div id="statusBody" class="text-sm"></div>
  </section>

  <footer class="text-xs text-steel-400/70 text-center mt-10">
    <p>Pipeline: pypdf · FastAPI · Tailwind CSS. Output file: <span class="font-mono">{stem}_print_HHMMSS_YYYYMMDD.pdf</span></p>
  </footer>
</div>

<script>
(() => {
  'use strict';
  const MAX_FILES = 20;
  const MAX_BYTES = 100 * 1024 * 1024;

  const dropzone  = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const fileList  = document.getElementById('fileList');
  const processBtn= document.getElementById('processBtn');
  const clearBtn  = document.getElementById('clearBtn');
  const statusPanel = document.getElementById('statusPanel');
  const statusBody  = document.getElementById('statusBody');

  /** @type {File[]} */
  let selected = [];

  const fmtBytes = (n) => {
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
    return (n / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const escapeHtml = (s) => s.replace(/&/g,'&amp;').replace(/</g,'&lt;')
                              .replace(/>/g,'&gt;').replace(/"/g,'&quot;');

  function renderList() {
    fileList.innerHTML = '';
    if (selected.length === 0) {
      fileList.classList.add('hidden');
      processBtn.disabled = true;
      clearBtn.disabled = true;
      return;
    }
    fileList.classList.remove('hidden');
    processBtn.disabled = false;
    clearBtn.disabled = false;

    selected.forEach((f) => {
      const row = document.createElement('div');
      row.className = 'flex items-center justify-between gap-3 rounded-lg border border-ink-600 bg-ink-800/60 px-3 py-2';
      row.innerHTML =
        '<div class="min-w-0 flex items-center gap-2">' +
          '<svg class="w-4 h-4 shrink-0 text-signal-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">' +
            '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z"/><path d="M14 2v6h6"/>' +
          '</svg>' +
          '<span class="truncate text-sm text-steel-200">' + escapeHtml(f.name) + '</span>' +
        '</div>' +
        '<span class="shrink-0 text-xs text-steel-400 font-mono">' + fmtBytes(f.size) + '</span>';
      fileList.appendChild(row);
    });
  }

  function toast(msg, kind) {
    statusPanel.classList.remove('hidden');
    const color = kind === 'err' ? 'text-red-400' : kind === 'warn' ? 'text-amber-400' : 'text-steel-300';
    statusBody.innerHTML = '<p class="' + color + '">' + escapeHtml(msg) + '</p>';
  }

  function addFiles(list) {
    const incoming = Array.from(list);
    for (const f of incoming) {
      const isPdf = f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf');
      if (!isPdf) { toast('Skipped non-PDF: ' + f.name, 'warn'); continue; }
      if (f.size > MAX_BYTES) { toast('Too large (>100 MB): ' + f.name, 'warn'); continue; }
      if (selected.length >= MAX_FILES) { toast('Maximum ' + MAX_FILES + ' files.', 'warn'); break; }
      if (selected.some(s => s.name === f.name && s.size === f.size)) continue;
      selected.push(f);
    }
    renderList();
  }

  // ── Drag & drop ────────────────────────────────────────────────────────
  ['dragenter','dragover'].forEach(ev =>
    dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add('drop-active'); }));
  ['dragleave','drop'].forEach(ev =>
    dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove('drop-active'); }));
  dropzone.addEventListener('drop', e => addFiles(e.dataTransfer.files));
  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => { addFiles(fileInput.files); fileInput.value = ''; });

  clearBtn.addEventListener('click', () => { selected = []; renderList(); statusPanel.classList.add('hidden'); });

  // ── Process ────────────────────────────────────────────────────────────
  processBtn.addEventListener('click', async () => {
    if (selected.length === 0) return;
    processBtn.disabled = true;
    clearBtn.disabled = true;
    processBtn.textContent = 'Processing…';
    statusPanel.classList.remove('hidden');
    statusBody.innerHTML =
      '<div class="flex items-center gap-3 text-steel-300">' +
        '<div class="w-4 h-4 rounded-full shimmer"></div>' +
        '<span>Composing ' + selected.length + ' file(s)…</span>' +
      '</div>';

    const form = new FormData();
    selected.forEach(f => form.append('files', f, f.name));

    try {
      const res = await fetch('/api/process', { method: 'POST', body: form });
      if (!res.ok) {
        let detail = 'HTTP ' + res.status;
        try { const j = await res.json(); if (j.detail) detail = j.detail; } catch (_) {}
        throw new Error(detail);
      }
      const blob = await res.blob();
      const cd = res.headers.get('Content-Disposition') || '';
      const m = cd.match(/filename="?([^"]+)"?/);
      const outName = m ? m[1] : 'output.pdf';

      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = outName;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 4000);

      const failed = res.headers.get('X-Failed-Count');
      statusBody.innerHTML =
        '<div class="flex items-start gap-3">' +
          '<svg class="w-5 h-5 mt-0.5 text-emerald-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
            '<path d="M20 6 9 17l-5-5"/></svg>' +
          '<div><p class="text-white font-medium">Download started</p>' +
          '<p class="text-steel-400 mt-1">' + escapeHtml(outName) +
          (failed ? ' · ' + failed + ' file(s) failed' : '') + '</p></div>' +
        '</div>';
    } catch (err) {
      statusBody.innerHTML =
        '<div class="flex items-start gap-3">' +
          '<svg class="w-5 h-5 mt-0.5 text-red-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
            '<circle cx="12" cy="12" r="10"/><path d="M12 8v4m0 4h.01"/></svg>' +
          '<div><p class="text-white font-medium">Processing failed</p>' +
          '<p class="text-red-400 mt-1">' + escapeHtml(String(err.message || err)) + '</p></div>' +
        '</div>';
    } finally {
      processBtn.disabled = false;
      clearBtn.disabled = false;
      processBtn.textContent = 'Compose PDFs';
    }
  });
})();
</script>
</body>
</html>
"""


# ═════════════════════════════════ Entry point ═══════════════════════════

def _probe_port(host: str, port: int) -> bool:
    """
    Return ``True`` iff ``(host, port)`` can be bound right now.

    ``SO_REUSEADDR`` is deliberately *not* set: on Windows it lets a probe
    bind over an active listener and report a false positive, which is
    exactly the failure mode that produced the original WinError 10048.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def _find_free_port(
    host: str,
    preferred: int,
    attempts: int = PORT_SCAN_ATTEMPTS,
) -> int:
    """
    Return ``preferred`` if it is bindable on ``host``; otherwise scan
    upward for the next free port.  Raises ``SystemExit`` if the whole
    range is exhausted.
    """
    for offset in range(attempts):
        candidate = preferred + offset
        if candidate > MAX_PORT:
            break
        if _probe_port(host, candidate):
            return candidate
    raise SystemExit(
        ERR_NO_FREE_PORT.format(
            start=preferred,
            end=min(preferred + attempts - 1, MAX_PORT),
            host=host,
        )
    )


def _load_server_config() -> tuple[str, int]:
    """
    Read HOST / PORT from the environment, validate them, and resolve a
    free port.  Defaults are loopback-only for safety.
    """
    host = os.environ.get("HOST", DEFAULT_HOST).strip() or DEFAULT_HOST

    raw_port = os.environ.get("PORT", str(DEFAULT_PORT)).strip()
    try:
        preferred_port = int(raw_port)
    except ValueError as exc:
        raise SystemExit(ERR_INVALID_PORT.format(value=raw_port)) from exc
    if not (1 <= preferred_port <= MAX_PORT):
        raise SystemExit(ERR_PORT_RANGE.format(port=preferred_port))

    actual_port = _find_free_port(host, preferred_port)
    if actual_port != preferred_port:
        print(
            f"[info] Port {preferred_port} is busy; using {actual_port} instead."
        )
    return host, actual_port


def _display_host(host: str) -> str:
    """Human-friendly host for the startup banner."""
    return "localhost" if host in {"0.0.0.0", "::", ""} else host


def main() -> None:
    import uvicorn

    host, port = _load_server_config()
    print(f"\n  ➜  Open: http://{_display_host(host)}:{port}\n")
    try:
        uvicorn.run(app, host=host, port=port, log_level=LOG_LEVEL)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
