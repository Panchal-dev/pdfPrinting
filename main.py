#!/usr/bin/env python3
"""
PDF Interleave + Rotate + 2-Up Composer — Web Edition
=====================================================

Single-file FastAPI application that exposes the PDF pipeline
(interleave A1,B1,A2,B2… → rotate → N-up sheet) as a browser-based
upload/download tool with a dark Tailwind UI and an **Advanced Settings**
panel that controls every stage of the pipeline.

Architecture
------------
  • The core PDF transformation is a pure function on bytes, fully
    decoupled from the HTTP layer, so it stays unit-testable and could be
    reused by a CLI or worker without modification.
  • The HTTP layer (FastAPI) handles multipart upload, form parsing,
    size caps, path sanitisation, and response streaming.
  • The presentation layer is one inline HTML document using Tailwind CSS
    (CDN) with a dark-first palette. No build step, no Node toolchain.

Quality / compression notes
---------------------------
  • Embedded images and fonts inside cloned pages are reused **by object
    reference** — pypdf never re-encodes them, so pixel data survives
    byte-for-byte.
  • ``output_compression`` lets you pick the strategy for page content
    streams:
        - ``preserve`` (default): keep whatever filters were present.
        - ``deflate``: force lossless FlateDecode (smaller files).
        - ``none``: fully **uncompressed** streams (largest files,
          readable without any decompression).
  • ``allow_upscale=False`` (default) prevents blurry enlargements of
    small source pages — a common edge case.

Pipeline invariants
-------------------
  • ``copy.copy`` precedes every ``rotate`` because ``PageObject.rotate``
    mutates the page in place; cloning first keeps copies independent.
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
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Any, Final, Mapping, Sequence

import starlette.status as _http_status
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pypdf import PageObject, PdfReader, PdfWriter, Transformation
from pypdf.generic import NameObject
from starlette.concurrency import run_in_threadpool


# ═════════════════════════════════ Constants ═════════════════════════════

# ── Unit conversion (PDF points; 1 pt = 1/72 inch; 1 in = 25.4 mm) ───────
PT_PER_INCH: Final[float] = 72.0
MM_PER_INCH: Final[float] = 25.4

# ── Sheet geometry (PDF points, portrait) ────────────────────────────────
# Reference: ISO 216 (A-series) and ANSI/Letter/Legal/Tabloid.
SHEET_SIZES_PORTRAIT_PT: Final[dict[str, tuple[float, float]]] = {
    "A5":      (419.528, 595.276),
    "A4":      (595.276, 841.890),
    "A3":      (841.890, 1190.551),
    "LETTER":  (612.0,   792.0),
    "LEGAL":   (612.0,  1008.0),
    "TABLOID": (792.0,  1224.0),
}
DEFAULT_SHEET_SIZE: Final[str] = "A4"

# ── Rotation ─────────────────────────────────────────────────────────────
# Value → pypdf rotate() argument (degrees; negative = counter-clockwise).
ROTATION_BY_NAME: Final[dict[str, int]] = {
    "left":  -90,
    "right":  90,
    "180":   180,
    "none":    0,
}
DEFAULT_ROTATION: Final[str] = "left"

# ── Orientation ──────────────────────────────────────────────────────────
ORIENTATION_CHOICES: Final[frozenset[str]] = frozenset(
    {"portrait", "landscape", "auto"}
)
DEFAULT_ORIENTATION: Final[str] = "landscape"

# ── Fit mode ─────────────────────────────────────────────────────────────
FIT_MODE_CHOICES: Final[frozenset[str]] = frozenset(
    {"fit", "stretch", "actual"}
)
DEFAULT_FIT_MODE: Final[str] = "fit"

# ── Output stream compression ────────────────────────────────────────────
COMPRESSION_CHOICES: Final[frozenset[str]] = frozenset(
    {"preserve", "deflate", "none"}
)
DEFAULT_COMPRESSION: Final[str] = "preserve"

# ── Layout constraints ───────────────────────────────────────────────────
SUPPORTED_PAGES_PER_SHEET: Final[frozenset[int]] = frozenset({1, 2, 4})
DEFAULT_PAGES_PER_SHEET: Final[int] = 2
MAX_MARGIN_MM: Final[float] = 25.0
MAX_GUTTER_MM: Final[float] = 25.0

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
class ComposerSettings:
    """
    Immutable container for every tunable in the pipeline.

    Instantiate via :func:`build_settings` so raw, untrusted form values
    are always coerced and clamped before reaching the pipeline.
    """

    rotation: str = DEFAULT_ROTATION
    interleave: bool = True
    pages_per_sheet: int = DEFAULT_PAGES_PER_SHEET
    sheet_size: str = DEFAULT_SHEET_SIZE
    sheet_orientation: str = DEFAULT_ORIENTATION
    fit_mode: str = DEFAULT_FIT_MODE
    margin_mm: float = 0.0
    gutter_mm: float = 0.0
    allow_upscale: bool = False
    output_compression: str = DEFAULT_COMPRESSION
    preserve_metadata: bool = True

    @property
    def rotation_degrees(self) -> int:
        return ROTATION_BY_NAME[self.rotation]


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


# ═════════════════════════════ Settings validation ═══════════════════════

def _parse_bool(value: Any, default: bool) -> bool:
    """Accept common truthy/falsey strings coming from HTML forms."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(
    value: Any, default: float, low: float, high: float
) -> float:
    """Coerce to float and clamp to ``[low, high]``; fall back on failure."""
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return default
    return max(low, min(high, parsed))


def _canonical_choice(
    value: Any, choices: Mapping[str, Any] | frozenset[str], default: str
) -> str:
    """Return the lower-cased value iff it is a member of ``choices``."""
    if value is None:
        return default
    candidate = str(value).strip().lower()
    if candidate in choices:
        return candidate
    return default


def _parse_pages_per_sheet(value: Any, default: int) -> int:
    """Accept only the small set of layouts the grid builder supports."""
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed in SUPPORTED_PAGES_PER_SHEET else default


def build_settings(raw: Mapping[str, Any]) -> ComposerSettings:
    """
    Build a validated :class:`ComposerSettings` from a raw mapping.

    Missing, malformed, or out-of-range values fall back to documented
    safe defaults so a single bad form field never aborts the request.
    """
    defaults = ComposerSettings()
    return ComposerSettings(
        rotation=_canonical_choice(
            raw.get("rotation"), ROTATION_BY_NAME, defaults.rotation
        ),
        interleave=_parse_bool(raw.get("interleave"), defaults.interleave),
        pages_per_sheet=_parse_pages_per_sheet(
            raw.get("pages_per_sheet"), defaults.pages_per_sheet
        ),
        sheet_size=_canonical_choice(
            raw.get("sheet_size"), SHEET_SIZES_PORTRAIT_PT, defaults.sheet_size
        ),
        sheet_orientation=_canonical_choice(
            raw.get("sheet_orientation"),
            ORIENTATION_CHOICES,
            defaults.sheet_orientation,
        ),
        fit_mode=_canonical_choice(
            raw.get("fit_mode"), FIT_MODE_CHOICES, defaults.fit_mode
        ),
        margin_mm=_parse_float(
            raw.get("margin_mm"), defaults.margin_mm, 0.0, MAX_MARGIN_MM
        ),
        gutter_mm=_parse_float(
            raw.get("gutter_mm"), defaults.gutter_mm, 0.0, MAX_GUTTER_MM
        ),
        allow_upscale=_parse_bool(
            raw.get("allow_upscale"), defaults.allow_upscale
        ),
        output_compression=_canonical_choice(
            raw.get("output_compression"),
            COMPRESSION_CHOICES,
            defaults.output_compression,
        ),
        preserve_metadata=_parse_bool(
            raw.get("preserve_metadata"), defaults.preserve_metadata
        ),
    )


# ═════════════════════════════════ PDF pipeline ══════════════════════════

def _mm_to_pt(millimetres: float) -> float:
    """Convert millimetres to PDF points."""
    return millimetres * PT_PER_INCH / MM_PER_INCH


def _grid_columns_rows(
    pages_per_sheet: int, sheet_w: float, sheet_h: float
) -> tuple[int, int]:
    """
    Return ``(columns, rows)`` for the requested pages-per-sheet count.

    Orientation-aware: a landscape sheet prefers side-by-side layout,
    while a portrait sheet prefers a vertical stack.
    """
    if pages_per_sheet == 1:
        return 1, 1
    if pages_per_sheet == 2:
        return (2, 1) if sheet_w >= sheet_h else (1, 2)
    if pages_per_sheet == 4:
        return 2, 2
    raise ValueError(f"Unsupported pages_per_sheet: {pages_per_sheet}")


def _resolve_sheet_size(settings: ComposerSettings) -> tuple[float, float]:
    """Return the effective ``(width, height)`` in points for a sheet."""
    base_w, base_h = SHEET_SIZES_PORTRAIT_PT[settings.sheet_size]

    orientation = settings.sheet_orientation
    if orientation == "auto":
        # Landscape when the natural layout is wider than it is tall.
        orientation = "landscape" if base_h > base_w else "portrait"

    if orientation == "landscape":
        # Ensure the long edge is horizontal.
        return (base_h, base_w) if base_h > base_w else (base_w, base_h)
    # Portrait: ensure the long edge is vertical.
    return (base_w, base_h) if base_h >= base_w else (base_h, base_w)


def _arrange_pages(
    source_pages: Sequence[PageObject],
    settings: ComposerSettings,
) -> list[PageObject]:
    """
    Produce the flat list of (cloned, optionally rotated) pages that will
    be laid out on the sheets.

    ``interleave=True`` preserves the original behaviour: every source
    page is duplicated ``pages_per_sheet`` times so the same page can be
    placed in every slot of its sheet.  ``interleave=False`` emits each
    page exactly once, so successive distinct pages share a sheet.
    """
    rotation = settings.rotation_degrees
    copies_per_page = settings.pages_per_sheet if settings.interleave else 1

    arranged: list[PageObject] = []
    for page in source_pages:
        for _ in range(copies_per_page):
            # copy.copy must precede rotate: rotate() mutates in place.
            clone = copy.copy(page)
            if rotation:
                clone.rotate(rotation)
            arranged.append(clone)
    return arranged


def _fit_scale(
    src_w: float,
    src_h: float,
    slot_w: float,
    slot_h: float,
    allow_upscale: bool,
) -> float:
    """
    Uniform, aspect-preserving scale that fits the source into the slot.

    When ``allow_upscale`` is False, the scale is additionally capped at
    1.0 so small source pages are never enlarged (which would blur raster
    content); they are simply centred in the slot instead.
    """
    if src_w <= 0.0 or src_h <= 0.0:
        return 1.0
    scale = min(slot_w / src_w, slot_h / src_h)
    if not allow_upscale:
        scale = min(scale, 1.0)
    return scale


def _compute_scale(
    src_w: float, src_h: float, slot_w: float, slot_h: float,
    settings: ComposerSettings,
) -> tuple[float, float]:
    """Return the (scale_x, scale_y) pair for the chosen fit mode."""
    if settings.fit_mode == "actual":
        return 1.0, 1.0
    if settings.fit_mode == "stretch":
        sx = slot_w / src_w if src_w > 0 else 1.0
        sy = slot_h / src_h if src_h > 0 else 1.0
        return sx, sy
    # "fit" — uniform, aspect-preserving.
    scale = _fit_scale(src_w, src_h, slot_w, slot_h, settings.allow_upscale)
    return scale, scale


def _place_page(
    sheet: PageObject,
    page: PageObject,
    slot_w: float,
    slot_h: float,
    x0: float,
    y0: float,
    settings: ComposerSettings,
) -> None:
    """
    Scale and centre ``page`` inside the slot whose bottom-left is ``(x0, y0)``.

    Embedded images and fonts are referenced, not re-encoded, so no
    pixel-level degradation occurs here.
    """
    src_w = float(page.mediabox.width)
    src_h = float(page.mediabox.height)

    scale_x, scale_y = _compute_scale(src_w, src_h, slot_w, slot_h, settings)
    new_w = src_w * scale_x
    new_h = src_h * scale_y

    translate_x = x0 + (slot_w - new_w) / 2
    translate_y = y0 + (slot_h - new_h) / 2

    sheet.merge_transformed_page(
        page,
        Transformation()
        .scale(scale_x, scale_y)
        .translate(translate_x, translate_y),
    )


def _compose_sheets(
    pages: Sequence[PageObject], settings: ComposerSettings
) -> tuple[PdfWriter, int]:
    """
    Compose ``pages`` into multi-up sheets according to ``settings``.

    Returns the populated writer and the number of sheets produced.
    """
    sheet_w, sheet_h = _resolve_sheet_size(settings)
    columns, rows = _grid_columns_rows(
        settings.pages_per_sheet, sheet_w, sheet_h
    )
    pages_per_sheet = columns * rows

    margin_pt = _mm_to_pt(settings.margin_mm)
    gutter_pt = _mm_to_pt(settings.gutter_mm)

    inner_w = sheet_w - 2 * margin_pt
    inner_h = sheet_h - 2 * margin_pt

    # Guard against user-supplied margins that would collapse the canvas.
    if inner_w <= 0 or inner_h <= 0:
        margin_pt = 0.0
        gutter_pt = 0.0
        inner_w = sheet_w
        inner_h = sheet_h

    slot_w = (inner_w - (columns - 1) * gutter_pt) / columns
    slot_h = (inner_h - (rows - 1) * gutter_pt) / rows

    writer = PdfWriter()
    n_pages = len(pages)
    n_sheets = 0

    for start in range(0, n_pages, pages_per_sheet):
        sheet = PageObject.create_blank_page(width=sheet_w, height=sheet_h)
        chunk = pages[start : start + pages_per_sheet]

        for index, page in enumerate(chunk):
            col = index % columns
            row = index // columns

            x0 = margin_pt + col * (slot_w + gutter_pt)
            # Layout is top-to-bottom, but PDF origin is bottom-left.
            y0 = (
                sheet_h
                - margin_pt
                - row * (slot_h + gutter_pt)
                - slot_h
            )
            _place_page(sheet, page, slot_w, slot_h, x0, y0, settings)

        writer.add_page(sheet)
        n_sheets += 1

    return writer, n_sheets


def _reencode_stream(
    stream_obj: Any, target_filter: str | None
) -> None:
    """
    Rewrite a single stream's bytes with the requested /Filter.

    ``target_filter=None`` means "no filter" (uncompressed).  Failures are
    treated as non-fatal: the stream keeps whatever encoding it had, so a
    best-effort guarantee rather than a hard failure.
    """
    try:
        raw = stream_obj.get_data()
    except Exception:
        return

    try:
        # Clear the existing filter before swapping bytes so the reader
        # never tries to decompress content that is now raw.
        if "/Filter" in stream_obj:
            del stream_obj["/Filter"]
        stream_obj.set_data(raw)
    except Exception:
        return

    if target_filter is not None:
        try:
            stream_obj[NameObject("/Filter")] = NameObject(target_filter)
        except Exception:
            pass


def _apply_output_compression(writer: PdfWriter, mode: str) -> None:
    """
    Apply the requested compression strategy to every page content stream.

    Only page ``/Contents`` streams are touched; embedded images and
    fonts keep their original encoding so no pixel data is ever altered.
    """
    if mode == "preserve":
        return

    target_filter = "/FlateDecode" if mode == "deflate" else None

    for page in writer.pages:
        try:
            contents = page.get_contents()
        except Exception:
            continue
        if contents is None:
            continue

        if hasattr(contents, "get_data"):
            streams = [contents]
        else:
            try:
                streams = list(contents)
            except TypeError:
                continue

        for stream_obj in streams:
            _reencode_stream(stream_obj, target_filter)


def _apply_metadata(
    writer: PdfWriter, reader: PdfReader, settings: ComposerSettings
) -> None:
    """Copy document metadata from the source into the composed output."""
    if not settings.preserve_metadata:
        return
    try:
        source = reader.metadata
    except Exception:
        return
    if not source:
        return
    try:
        writer.add_metadata(dict(source))
    except Exception:
        # Metadata is best-effort; a malformed entry must not abort the run.
        return


def process_pdf_bytes(
    pdf_bytes: bytes, stem: str, settings: ComposerSettings
) -> ProcessedPdf:
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

    arranged = _arrange_pages(reader.pages, settings)
    writer, n_sheets = _compose_sheets(arranged, settings)
    _apply_metadata(writer, reader, settings)
    _apply_output_compression(writer, settings.output_compression)

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

def _process_one(
    file: UploadFile, data: bytes, settings: ComposerSettings
) -> ProcessedPdf:
    """Run the pipeline for a single validated upload."""
    stem = _safe_stem(file.filename)
    return process_pdf_bytes(data, stem, settings)


def process_uploads(
    files: Sequence[UploadFile],
    payloads: Sequence[bytes],
    settings: ComposerSettings,
) -> BatchResult:
    """
    Process every upload, collecting per-file failures instead of aborting
    the whole request on the first bad input.
    """
    processed: list[ProcessedPdf] = []
    failures: list[str] = []

    for file, data in zip(files, payloads):
        try:
            processed.append(_process_one(file, data, settings))
        except ValueError as exc:
            failures.append(f"{file.filename or '<unnamed>'}: {exc}")

    return BatchResult(files=tuple(processed), failures=tuple(failures))


def _build_response(batch: BatchResult, settings: ComposerSettings) -> Response:
    """Return a single PDF, a ZIP of many, or a 422 when everything failed."""
    if not batch.files:
        detail = "; ".join(batch.failures) or ERR_NO_FILES
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE, detail=detail
        )

    common_headers = {
        "X-Pages-Per-Sheet": str(settings.pages_per_sheet),
        "X-Sheet-Size": settings.sheet_size,
        "X-Sheet-Orientation": settings.sheet_orientation,
        "X-Compression": settings.output_compression,
    }

    if len(batch.files) == 1:
        only = batch.files[0]
        headers = {
            "Content-Disposition": (
                f'attachment; filename="{only.output_name}"'
            ),
            "X-Input-Pages": str(only.input_pages),
            "X-Output-Sheets": str(only.output_sheets),
            **common_headers,
        }
        return Response(
            content=only.content,
            media_type="application/pdf",
            headers=headers,
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
        **common_headers,
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
    title="PDF Interleave + Rotate + N-Up Composer",
    description=(
        "Interleave a PDF, rotate every page (default 90° counter-"
        "clockwise), and lay the result out as multi-up sheets with "
        "configurable size, orientation, fit mode, margins and stream "
        "compression."
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
    rotation: Annotated[str, Form()] = DEFAULT_ROTATION,
    interleave: Annotated[str, Form()] = "true",
    pages_per_sheet: Annotated[str, Form()] = str(DEFAULT_PAGES_PER_SHEET),
    sheet_size: Annotated[str, Form()] = DEFAULT_SHEET_SIZE,
    sheet_orientation: Annotated[str, Form()] = DEFAULT_ORIENTATION,
    fit_mode: Annotated[str, Form()] = DEFAULT_FIT_MODE,
    margin_mm: Annotated[str, Form()] = "0",
    gutter_mm: Annotated[str, Form()] = "0",
    allow_upscale: Annotated[str, Form()] = "false",
    output_compression: Annotated[str, Form()] = DEFAULT_COMPRESSION,
    preserve_metadata: Annotated[str, Form()] = "true",
) -> Response:
    """
    Accept one or more PDF uploads, run the pipeline using the supplied
    settings, and return either a single PDF or a ZIP archive containing
    every result.
    """
    if not files:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE, detail=ERR_NO_FILES
        )
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=HTTP_413_TOO_LARGE, detail=ERR_TOO_MANY_FILES
        )

    settings = build_settings(
        {
            "rotation": rotation,
            "interleave": interleave,
            "pages_per_sheet": pages_per_sheet,
            "sheet_size": sheet_size,
            "sheet_orientation": sheet_orientation,
            "fit_mode": fit_mode,
            "margin_mm": margin_mm,
            "gutter_mm": gutter_mm,
            "allow_upscale": allow_upscale,
            "output_compression": output_compression,
            "preserve_metadata": preserve_metadata,
        }
    )

    payloads: list[bytes] = []
    for file in files:
        data = await _read_upload_capped(file)
        _validate_upload_metadata(file, len(data))
        payloads.append(data)

    # pypdf is CPU-bound and synchronous; offload so the event loop stays
    # free to serve other requests.
    batch = await run_in_threadpool(
        process_uploads, files, payloads, settings
    )
    return _build_response(batch, settings)


# ═════════════════════════════════ Presentation ══════════════════════════

INDEX_HTML: Final[str] = r"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>PDF Composer — Interleave · Rotate · N-Up</title>
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

  /* Advanced settings panel styling */
  details.adv > summary { list-style: none; }
  details.adv > summary::-webkit-details-marker { display: none; }
  details.adv[open] > summary .chev { transform: rotate(90deg); }
  .chev { transition: transform .15s ease; }
  .field-label { font-size: .75rem; letter-spacing: .02em; text-transform: uppercase; }
  .field-input {
    background: rgba(11,15,20,.75); border: 1px solid #1f2b3a; color: #d7e1ea;
    border-radius: .5rem; padding: .45rem .65rem; font-size: .85rem; width: 100%;
  }
  .field-input:focus { outline: 2px solid #22d3ee; outline-offset: 0; border-color:#22d3ee; }
  .chk { accent-color: #22d3ee; }
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
      Interleave · Rotate · N-Up compose in one pass.
      Everything runs in-memory; nothing is written to the server's disk.
      Embedded images and fonts are preserved byte-for-byte — the
      <span class="font-mono text-steel-300">output&nbsp;compression</span>
      setting only affects page content streams.
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

    <!-- ── Advanced settings ──────────────────────────────────────────── -->
    <details class="adv mt-5 rounded-xl border border-ink-600 bg-ink-800/60 overflow-hidden">
      <summary class="cursor-pointer select-none flex items-center gap-2 px-4 py-3 text-sm font-medium text-steel-300 hover:text-white transition-colors">
        <svg class="chev w-3.5 h-3.5 text-signal-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
          <path d="m9 6 6 6-6 6"/>
        </svg>
        Advanced Settings
        <span class="ml-auto text-xs text-steel-400 font-normal">rotation · layout · quality</span>
      </summary>

      <div class="border-t border-ink-600 px-4 py-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">

        <!-- Rotation -->
        <label class="block">
          <span class="field-label text-steel-400 block mb-1">Rotation</span>
          <select id="optRotation" class="field-input">
            <option value="left" selected>90° Left (CCW)</option>
            <option value="right">90° Right (CW)</option>
            <option value="180">180°</option>
            <option value="none">None</option>
          </select>
        </label>

        <!-- Pages per sheet -->
        <label class="block">
          <span class="field-label text-steel-400 block mb-1">Pages per sheet</span>
          <select id="optPagesPerSheet" class="field-input">
            <option value="2" selected>2 (2-up)</option>
            <option value="1">1 (single)</option>
            <option value="4">4 (2×2 grid)</option>
          </select>
        </label>

        <!-- Sheet size -->
        <label class="block">
          <span class="field-label text-steel-400 block mb-1">Sheet size</span>
          <select id="optSheetSize" class="field-input">
            <option value="A4" selected>A4</option>
            <option value="A3">A3</option>
            <option value="A5">A5</option>
            <option value="LETTER">Letter</option>
            <option value="LEGAL">Legal</option>
            <option value="TABLOID">Tabloid</option>
          </select>
        </label>

        <!-- Orientation -->
        <label class="block">
          <span class="field-label text-steel-400 block mb-1">Orientation</span>
          <select id="optOrientation" class="field-input">
            <option value="landscape" selected>Landscape</option>
            <option value="portrait">Portrait</option>
            <option value="auto">Auto</option>
          </select>
        </label>

        <!-- Fit mode -->
        <label class="block">
          <span class="field-label text-steel-400 block mb-1">Fit mode</span>
          <select id="optFitMode" class="field-input">
            <option value="fit" selected>Fit (aspect-preserving)</option>
            <option value="stretch">Stretch to fill slot</option>
            <option value="actual">Actual size (no scaling)</option>
          </select>
        </label>

        <!-- Output compression -->
        <label class="block">
          <span class="field-label text-steel-400 block mb-1">Output compression</span>
          <select id="optCompression" class="field-input">
            <option value="preserve" selected>Preserve original</option>
            <option value="deflate">Deflate (lossless, smaller)</option>
            <option value="none">None (uncompressed, largest)</option>
          </select>
        </label>

        <!-- Margin -->
        <label class="block">
          <span class="field-label text-steel-400 block mb-1">Margin (mm)</span>
          <input id="optMargin" type="number" min="0" max="25" step="0.5" value="0" class="field-input" />
        </label>

        <!-- Gutter -->
        <label class="block">
          <span class="field-label text-steel-400 block mb-1">Gutter between slots (mm)</span>
          <input id="optGutter" type="number" min="0" max="25" step="0.5" value="0" class="field-input" />
        </label>

        <!-- Toggles -->
        <div class="space-y-3 pt-2 sm:pt-0">
          <label class="flex items-center gap-2 text-sm text-steel-300">
            <input id="optInterleave" type="checkbox" checked class="chk w-4 h-4" />
            Duplicate pages (interleave)
          </label>
          <label class="flex items-center gap-2 text-sm text-steel-300">
            <input id="optAllowUpscale" type="checkbox" class="chk w-4 h-4" />
            Allow upscaling (may blur)
          </label>
          <label class="flex items-center gap-2 text-sm text-steel-300">
            <input id="optPreserveMeta" type="checkbox" checked class="chk w-4 h-4" />
            Preserve metadata
          </label>
        </div>

      </div>
    </details>

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

  const escapeHtml = (s) => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
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

  clearBtn.addEventListener('click', () => {
    selected = [];
    renderList();
    statusPanel.classList.add('hidden');
  });

  // ── Advanced settings collection ───────────────────────────────────────
  function collectSettings(form) {
    const val = (id) => document.getElementById(id);
    form.append('rotation',            val('optRotation').value);
    form.append('pages_per_sheet',     val('optPagesPerSheet').value);
    form.append('sheet_size',          val('optSheetSize').value);
    form.append('sheet_orientation',   val('optOrientation').value);
    form.append('fit_mode',            val('optFitMode').value);
    form.append('output_compression',  val('optCompression').value);
    form.append('margin_mm',           String(val('optMargin').value || '0'));
    form.append('gutter_mm',           String(val('optGutter').value || '0'));
    form.append('interleave',          val('optInterleave').checked ? 'true' : 'false');
    form.append('allow_upscale',       val('optAllowUpscale').checked ? 'true' : 'false');
    form.append('preserve_metadata',   val('optPreserveMeta').checked ? 'true' : 'false');
  }

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
    collectSettings(form);

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
      const pps = res.headers.get('X-Pages-Per-Sheet') || '';
      const size = res.headers.get('X-Sheet-Size') || '';
      const comp = res.headers.get('X-Compression') || '';
      const summary = [pps && pps + '-up', size, comp].filter(Boolean).join(' · ');

      statusBody.innerHTML =
        '<div class="flex items-start gap-3">' +
          '<svg class="w-5 h-5 mt-0.5 text-emerald-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
            '<path d="M20 6 9 17l-5-5"/></svg>' +
          '<div><p class="text-white font-medium">Download started</p>' +
          '<p class="text-steel-400 mt-1">' + escapeHtml(outName) +
          (summary ? ' · ' + escapeHtml(summary) : '') +
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
