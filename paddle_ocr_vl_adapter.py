"""PaddleOCR-VL-1.6 adapter for table_recognizer_gui.

The base GUI (``table_recognizer_gui.py``) talks to a PaddleOCR table-recognition
service over HTTP and expects the classic JSON payload (``tableRecResults`` /
``overall_ocr_res`` / ``pred_html``).

This module instead talks to a vLLM-served ``PaddleOCR-VL-1.6`` model through the
OpenAI-compatible ``/v1/chat/completions`` endpoint.  PaddleOCR-VL emits tables in
its own "OTSL" markup (``<fcel>`` ``<ecel>`` ``<nl>`` and the span tags ``<lcel>``
``<ucel>`` ``<xcel>``).  We convert OTSL -> HTML and let the base module's existing
HTML table parser turn that into ``ParsedTable`` objects, so every downstream
step (header detection, spreadsheet generation, …) is unchanged.

Contract discovered from the served model and PaddleX's PaddleOCR-VL pipeline:

* Endpoint:  POST {api_url}  (OpenAI ``/v1/chat/completions``)
* Model:     "PaddleOCR-VL-1.6"
* Prompt:    "Table Recognition:"   (free form is also fine, but the exact
             pipeline prompt reliably triggers OTSL table output)
* Output:    a single OTSL string; convert with ``convert_otsl_to_html``.
* Pixel budget: the official pipeline uses min 112896 / max 1003520 px for table
             blocks.  We resize the rectified page so its long edge is capped to
             keep the request well within token limits.
"""

from __future__ import annotations

import base64
import html
import itertools
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import table_recognizer_gui as base

DEFAULT_API = "http://192.168.10.26:8003/v1/chat/completions"
MODEL_NAME = "PaddleOCR-VL-1.6"
TABLE_PROMPT = "Table Recognition:"

# Token budget used by PaddleX for PaddleOCR-VL (PADDLEOCR_VL_MAX_NEW_TOKENS).
MAX_NEW_TOKENS = 8192
MAX_IMAGE_LONG_EDGE = 2000
MIN_PIXELS = 112896
MAX_PIXELS = 1003520
TABLE_BLOCK_MIN_PIXELS = 112896
TABLE_BLOCK_MAX_PIXELS = 1003520


# --------------------------------------------------------------------------- #
# OTSL parser (ported from PaddleX `paddlex/inference/pipelines/paddleocr_vl`)
# --------------------------------------------------------------------------- #
OTSL_NL = "<nl>"
OTSL_FCEL = "<fcel>"
OTSL_ECEL = "<ecel>"
OTSL_LCEL = "<lcel>"
OTSL_UCEL = "<ucel>"
OTSL_XCEL = "<xcel>"
_OTSL_TAGS = (OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL)
_NON_CAPTURING_TAG_GROUP = "(?:<fcel>|<ecel>|<nl>|<lcel>|<ucel>|<xcel>)"
_OTSL_FIND_PATTERN = re.compile(
    f"{_NON_CAPTURING_TAG_GROUP}.*?(?={_NON_CAPTURING_TAG_GROUP}|$)", flags=re.DOTALL
)


def _otsl_extract_tokens_and_text(s: str):
    pattern = "(" + "|".join(_OTSL_TAGS) + ")"
    tokens = re.findall(pattern, s)
    text_parts = [part for part in re.split(pattern, s) if part.strip()]
    return tokens, text_parts


def _otsl_parse_texts(texts, tokens):
    split_word = OTSL_NL
    split_row_tokens = [
        list(group)
        for is_nl, group in itertools.groupby(tokens, lambda z: z == split_word)
        if not is_nl
    ]
    table_cells = []
    r_idx = 0
    c_idx = 0

    # Ensure matrix completeness so ragged tables parse deterministically.
    if split_row_tokens:
        max_cols = max(len(row) for row in split_row_tokens)
        for row in split_row_tokens:
            while len(row) < max_cols:
                row.append(OTSL_ECEL)
        new_texts = []
        text_idx = 0
        for row in split_row_tokens:
            for token in row:
                new_texts.append(token)
                if text_idx < len(texts) and texts[text_idx] == token:
                    text_idx += 1
                    if text_idx < len(texts) and texts[text_idx] not in _OTSL_TAGS:
                        new_texts.append(texts[text_idx])
                        text_idx += 1
            new_texts.append(OTSL_NL)
            if text_idx < len(texts) and texts[text_idx] == OTSL_NL:
                text_idx += 1
        texts = new_texts

    def count_right(tokens, c_idx, r_idx, which_tokens):
        span = 0
        c_iter = c_idx
        while tokens[r_idx][c_iter] in which_tokens:
            c_iter += 1
            span += 1
            if c_iter >= len(tokens[r_idx]):
                return span
        return span

    def count_down(tokens, c_idx, r_idx, which_tokens):
        span = 0
        r_iter = r_idx
        while tokens[r_iter][c_idx] in which_tokens:
            r_iter += 1
            span += 1
            if r_iter >= len(tokens):
                return span
        return span

    for i, text in enumerate(texts):
        cell_text = ""
        if text in (OTSL_FCEL, OTSL_ECEL):
            row_span = 1
            col_span = 1
            right_offset = 1
            if text != OTSL_ECEL:
                cell_text = texts[i + 1]
                right_offset = 2
            next_right_cell = (
                texts[i + right_offset] if i + right_offset < len(texts) else ""
            )
            next_bottom_cell = ""
            if r_idx + 1 < len(split_row_tokens) and c_idx < len(
                split_row_tokens[r_idx + 1]
            ):
                next_bottom_cell = split_row_tokens[r_idx + 1][c_idx]
            if next_right_cell in (OTSL_LCEL, OTSL_XCEL):
                col_span += count_right(
                    split_row_tokens, c_idx + 1, r_idx, [OTSL_LCEL, OTSL_XCEL]
                )
            if next_bottom_cell in (OTSL_UCEL, OTSL_XCEL):
                row_span += count_down(
                    split_row_tokens, c_idx, r_idx + 1, [OTSL_UCEL, OTSL_XCEL]
                )
            table_cells.append(
                {
                    "text": cell_text.strip(),
                    "row_span": row_span,
                    "col_span": col_span,
                    "start_row": r_idx,
                    "end_row": r_idx + row_span,
                    "start_col": c_idx,
                    "end_col": c_idx + col_span,
                }
            )
        if text in (OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL):
            c_idx += 1
        if text == OTSL_NL:
            r_idx += 1
            c_idx = 0
    return table_cells, split_row_tokens


def _otsl_pad_to_sqr_v2(otsl_str: str) -> str:
    otsl_str = otsl_str.strip()
    if OTSL_NL not in otsl_str:
        return otsl_str + OTSL_NL
    lines = otsl_str.split(OTSL_NL)
    row_data = []
    for line in lines:
        if not line:
            continue
        raw_cells = _OTSL_FIND_PATTERN.findall(line)
        if not raw_cells:
            continue
        min_len = 0
        for i, cell_str in enumerate(raw_cells):
            if cell_str.startswith(OTSL_FCEL):
                min_len = i + 1
        row_data.append(
            {"raw_cells": raw_cells, "total_len": len(raw_cells), "min_len": min_len}
        )
    if not row_data:
        return OTSL_NL
    global_min_width = max(row["min_len"] for row in row_data)
    max_total_len = max(row["total_len"] for row in row_data)
    search_start = global_min_width
    search_end = max(global_min_width, max_total_len)
    min_total_cost = float("inf")
    optimal_width = search_end
    for width in range(search_start, search_end + 1):
        cost = sum(abs(row["total_len"] - width) for row in row_data)
        if cost < min_total_cost:
            min_total_cost = cost
            optimal_width = width
    repaired_lines = []
    for row in row_data:
        cells = row["raw_cells"]
        current_len = len(cells)
        if current_len > optimal_width:
            new_cells = cells[:optimal_width]
        else:
            new_cells = cells + [OTSL_ECEL] * (optimal_width - current_len)
        repaired_lines.append("".join(new_cells))
    return OTSL_NL.join(repaired_lines) + OTSL_NL


def _otsl_export_to_html(table_cells, nrows, ncols) -> str:
    if not table_cells:
        return ""
    grid = [[None] * ncols for _ in range(nrows)]
    for cell in table_cells:
        for i in range(min(cell["start_row"], nrows), min(cell["end_row"], nrows)):
            for j in range(min(cell["start_col"], ncols), min(cell["end_col"], ncols)):
                grid[i][j] = cell
    body = ""
    for i in range(nrows):
        body += "<tr>"
        for j in range(ncols):
            cell = grid[i][j]
            if cell is None:
                continue
            rowspan = cell["row_span"]
            colspan = cell["col_span"]
            if cell["start_row"] != i or cell["start_col"] != j:
                continue
            content = html.escape(cell["text"])
            opening = ""
            if rowspan > 1:
                opening += f' rowspan="{rowspan}"'
            if colspan > 1:
                opening += f' colspan="{colspan}"'
            body += f'<td{opening}>{content}</td>'
        body += "</tr>"
    return f"<table>{body}</table>"


def convert_otsl_to_html(otsl_content: str) -> str:
    """Convert an OTSL-v1.0 string to an HTML table."""
    if not otsl_content:
        return ""
    otsl_content = _otsl_pad_to_sqr_v2(otsl_content)
    tokens, mixed_texts = _otsl_extract_tokens_and_text(otsl_content)
    table_cells, split_row_tokens = _otsl_parse_texts(mixed_texts, tokens)
    nrows = len(split_row_tokens)
    ncols = max((len(row) for row in split_row_tokens), default=0)
    return _otsl_export_to_html(table_cells, nrows, ncols)


def _find_repeating_suffix(s: str, min_len: int = 8, min_repeats: int = 5):
    for i in range(len(s) // min_repeats, min_len - 1, -1):
        unit = s[-i:]
        if s.endswith(unit * min_repeats):
            count = 0
            temp_s = s
            while temp_s.endswith(unit):
                temp_s = temp_s[:-i]
                count += 1
            start_index = len(s) - count * i
            return s[:start_index], unit, count
    return None


def _find_shortest_repeating_substring(s: str):
    n = len(s)
    for i in range(1, n // 2 + 1):
        if n % i == 0:
            substring = s[:i]
            if substring * (n // i) == s:
                return substring
    return None


def truncate_repetitive_content(
    content: str,
    line_threshold: int = 10,
    char_threshold: int = 10,
    min_len: int = 10,
    min_count: int = 3000,
) -> str:
    """Detect and trim degenerately repeated model output."""
    if len(content) < min_count:
        return content
    stripped = content.strip()
    if not stripped:
        return content
    if "\n" not in stripped and len(stripped) > 100:
        match = _find_repeating_suffix(stripped, min_len=8, min_repeats=5)
        if match:
            prefix, unit, count = match
            if len(unit) * count > len(stripped) * 0.5:
                return prefix
    if "\n" not in stripped and len(stripped) > min_len:
        unit = _find_shortest_repeating_substring(stripped)
        if unit and len(stripped) // len(unit) >= char_threshold:
            return unit
    return content


# --------------------------------------------------------------------------- #
# Image prep
# --------------------------------------------------------------------------- #
def _fit_pixel_budget(image_bytes: bytes, max_pixels: int, max_edge: int) -> bytes:
    """Resize so the image fits the model's pixel / token budget."""
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解码预处理后的图片")
    height, width = image.shape[:2]
    limit = min(max_pixels, max_edge * max_edge)
    scale = min(1.0, (limit / (width * height)) ** 0.5, max_edge / max(width, height))
    if scale < 1.0:
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    ok, output = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("预处理图片编码失败")
    return output.tobytes()


# --------------------------------------------------------------------------- #
# API call
# --------------------------------------------------------------------------- #
def _request_otsl(image_bytes: bytes, api_url: str, timeout: int, prompt: str = TABLE_PROMPT) -> str:
    body = json.dumps(
        {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,"
                                + base64.b64encode(image_bytes).decode("ascii")
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"PaddleOCR-VL 服务返回 HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"无法连接 PaddleOCR-VL 服务：{exc}") from exc

    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("PaddleOCR-VL 服务未返回识别结果")
    content = (choices[0].get("message") or {}).get("content") or ""
    return truncate_repetitive_content(content)


# --------------------------------------------------------------------------- #
# Archive-form normalization
# --------------------------------------------------------------------------- #
# The photographed archive forms are a FIXED 14-column template.  PaddleOCR-VL
# tends to split the handwritten month (e.g. "1月"/"2月"/"5月") written in the
# 日期/档号 header zone into its OWN column right after 档号, which shifts every
# later column and makes the column count differ between pages (15 vs 13 vs 12).
# It also occasionally drops the trailing "挂接条目/装盒" columns.  These helpers
# strip the spurious month column and re-align every page onto the canonical
# ARCHIVE_TABLE_HEADERS grid so all sheets line up.
_ARCHIVE_SIGNATURES = (
    "档号", "装订", "编制页", "条目", "打印卷", "文档扫", "图纸扫", "挂接", "装盒",
)


def _is_archive_header(row: list[str]) -> bool:
    joined = re.sub(r"\s+", "", "".join(str(cell) for cell in row))
    return sum(signature in joined for signature in _ARCHIVE_SIGNATURES) >= 4


def _find_code_column(rows: list[list[str]]) -> int | None:
    """Return the index of the 档号 column, or None.

    Every archive row carries a file number (``800217-…``), so the column whose
    values overwhelmingly match that pattern is the reliable anchor — available
    even on continuation pages that have no header row.
    """
    width = max((len(row) for row in rows), default=0)
    best_col, best_hits = None, 0
    for col in range(width):
        hits = sum(
            1
            for row in rows
            if col < len(row) and re.match(r"^800217-", str(row[col]).strip())
        )
        if hits > best_hits:
            best_hits, best_col = hits, col
    return best_col if best_hits >= 3 else None


def _is_spurious_month(cell: object) -> bool:
    """Return whether a header-only cell is a handwritten month label."""
    text = re.sub(r"\s+", "", str(cell))
    return bool(re.fullmatch(r"(?:[1-9]|1[0-2])(?:月|��)", text))


def _row_code_column(row: list[str]) -> int | None:
    for col, cell in enumerate(row):
        if re.match(r"^800217-", str(cell).strip()):
            return col
    return None


def _restore_omitted_business_column(row: list[str]) -> None:
    """Restore a fully empty D/E column that OTSL occasionally omits.

    Codes beginning ``800217-0...`` belong to engineering/science archives and
    use canonical column E; the document-style codes beginning ``800217-2...``
    use column D.  In these forms D and E are mutually exclusive per data row.
    This gives us a safe way to restore the empty peer column when PaddleOCR-VL
    collapses it instead of emitting ``<ecel>``.
    """
    if len(row) < 5:
        return
    code = str(row[1]).strip()
    match = re.match(r"^800217-([02])", code)
    if not match:
        return

    if match.group(1) == "0":
        # Engineering/science record: D must be the empty peer of populated E.
        if str(row[3]).strip():
            row.insert(3, "")
    else:
        # Document/purchase/contract record: populated D is followed by empty E.
        if str(row[3]).strip() and str(row[4]).strip():
            row.insert(4, "")


def normalize_archive_table(table: base.ParsedTable) -> base.ParsedTable:
    """Re-align the archive form onto the canonical ARCHIVE_TABLE_HEADERS grid.

    PaddleOCR-VL emits a variable-width grid across pages: it splits the
    handwritten month into its own column, and on header-less continuation pages
    it drops the 日期 column and sometimes the trailing 挂接条目/装盒 columns.
    We anchor on the 档号 column (present in every row) and shift each page so
    档号 lands at canonical index 1, then pad/trim to the fixed width.
    """
    if not table.rows:
        return table
    rows = [list(str(cell) for cell in row) for row in table.rows]

    # Detect the archive form by content (档号 column) OR by header words, so
    # header-less continuation pages are normalized too.
    code_col = _find_code_column(rows)
    if code_col is None and not _is_archive_header(rows[0]):
        return table

    # Strip the handwritten-month column only on a genuine header row.  The old
    # "any short digit" rule also deleted ordinary continuation-page values such
    # as 1 and 9, shifting every later field to the left.
    has_archive_header = _is_archive_header(rows[0])
    width = max(len(row) for row in rows)
    strip = set()
    if has_archive_header:
        for col in range(2, min(4, width)):
            if col < len(rows[0]) and _is_spurious_month(rows[0][col]):
                strip.add(col)
    for col in sorted(strip, reverse=True):
        for row in rows:
            if col < len(row):
                del row[col]

    # Anchor each row independently.  Continuation pages commonly omit the empty
    # 日期 cell, while a dated row on that same page keeps it; a page-wide shift
    # therefore misaligned exactly those boundary rows.
    for row in rows:
        row_code_col = _row_code_column(row)
        if row_code_col is None:
            continue
        shift = 1 - row_code_col
        if shift > 0:
            row[:0] = [""] * shift
        elif shift < 0:
            del row[:-shift]
        _restore_omitted_business_column(row)

    # Re-align width to the canonical template, padding missing trailing cols.
    canonical_cols = len(base.ARCHIVE_TABLE_HEADERS)
    for row in rows:
        if len(row) < canonical_cols:
            row.extend([""] * (canonical_cols - len(row)))
        elif len(row) > canonical_cols:
            del row[canonical_cols:]

    table.rows = rows
    header_rows = 1 if table.header_rows else base.detect_header_rows(rows)
    table.header_rows = header_rows
    if header_rows and rows:
        rows[0] = list(base.ARCHIVE_TABLE_HEADERS)
    return table


# --------------------------------------------------------------------------- #
# Public adapter
# --------------------------------------------------------------------------- #
def _finalize_tables(parsed: list[base.ParsedTable], source_name: str) -> list[base.ParsedTable]:
    tables: list[base.ParsedTable] = []
    for table in parsed:
        table.source = source_name
        table.header_rows = base.detect_header_rows(table.rows)
        if table.header_rows and table.rows:
            table.rows[0] = base.normalize_known_header(table.rows[0])  # type: ignore[index]
        normalize_archive_table(table)
        tables.append(table)
    return tables


def recognize_with_paddle_ocr_vl(
    path, api_url: str, timeout: int = 300
) -> list[base.ParsedTable]:
    path = Path(path)
    # Use the preprocessed image (from the "图片预处理" tab) if present, else the
    # default rectified photo.
    rectified = (
        base.get_preprocessed(path)
        if base.get_preprocessed(path) is not None
        else base.rectify_table_image(path)
    )
    image_bytes = _fit_pixel_budget(
        rectified, min(MAX_PIXELS, TABLE_BLOCK_MAX_PIXELS), MAX_IMAGE_LONG_EDGE
    )
    otsl = _request_otsl(image_bytes, api_url, timeout)
    html_str = convert_otsl_to_html(otsl)

    parsed = base.parse_html_tables(html_str)
    tables = _finalize_tables(parsed, path.name)
    if not tables:
        raise RuntimeError(f"PaddleOCR-VL 没有解析出表格。OTSL 原文：{otsl[:500]}")
    return tables


def _dump_raw(path, api_url: str, timeout: int) -> str:
    path = Path(path)
    rectified = (
        base.get_preprocessed(path)
        if base.get_preprocessed(path) is not None
        else base.rectify_table_image(path)
    )
    image_bytes = _fit_pixel_budget(
        rectified, min(MAX_PIXELS, TABLE_BLOCK_MAX_PIXELS), MAX_IMAGE_LONG_EDGE
    )
    return _request_otsl(image_bytes, api_url, timeout)


def recognize_with_paddle_ocr_vl_debug(
    path, api_url: str, timeout: int = 300
) -> tuple[list[base.ParsedTable], str, str]:
    """Like ``recognize_with_paddle_ocr_vl`` but also returns the raw OTSL and HTML.

    Useful for CLI inspection/tests; the GUI uses the table-only variant.
    """
    path = Path(path)
    rectified = (
        base.get_preprocessed(path)
        if base.get_preprocessed(path) is not None
        else base.rectify_table_image(path)
    )
    image_bytes = _fit_pixel_budget(
        rectified, min(MAX_PIXELS, TABLE_BLOCK_MAX_PIXELS), MAX_IMAGE_LONG_EDGE
    )
    otsl = _request_otsl(image_bytes, api_url, timeout)
    html_str = convert_otsl_to_html(otsl)
    parsed = base.parse_html_tables(html_str)
    tables = _finalize_tables(parsed, path.name)
    if not tables:
        raise RuntimeError(f"PaddleOCR-VL 没有解析出表格。OTSL 原文：{otsl[:500]}")
    return tables, otsl, html_str
