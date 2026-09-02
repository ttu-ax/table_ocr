from __future__ import annotations

import io
import re
import threading
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, "MixTeXModel"] = {}


def _pad_image(image: Image.Image, size: tuple[int, int] = (448, 448)) -> Image.Image:
    image = image.convert("RGB")
    target_w, target_h = size
    canvas = Image.new("RGB", size, "white")
    width, height = image.size
    scale = min(target_w / width, target_h / height, 1.0)
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas.paste(resized, ((target_w - resized.width) // 2, (target_h - resized.height) // 2))
    return canvas


def _has_long_repetition(text: str, repeats: int = 18) -> bool:
    tail = text[-240:]
    for size in range(1, min(24, len(tail) // repeats) + 1):
        if tail.endswith(tail[-size:] * repeats):
            return True
    return False


class MixTeXModel:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir.resolve()
        if not (self.model_dir / "encoder_model.onnx").is_file():
            raise FileNotFoundError(f"找不到 MixTeX 编码器：{self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
        self.processor = AutoImageProcessor.from_pretrained(str(self.model_dir), local_files_only=True)
        providers = ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.encoder = ort.InferenceSession(
            str(self.model_dir / "encoder_model.onnx"), options, providers=providers
        )
        self.decoder = ort.InferenceSession(
            str(self.model_dir / "decoder_model_merged.onnx"), options, providers=providers
        )
        layer_numbers = {
            int(match.group(1))
            for item in self.decoder.get_inputs()
            if (match := re.match(r"past_key_values\.(\d+)\.(?:key|value)$", item.name))
        }
        self.num_layers = len(layer_numbers)
        if not self.num_layers:
            raise RuntimeError("无法从 MixTeX 解码器读取层数")

    def recognize(self, image: Image.Image, max_length: int = 512) -> str:
        image = _pad_image(image)
        pixels = self.processor(image, return_tensors="np").pixel_values
        encoder_out = self.encoder.run(None, {"pixel_values": pixels})[0]
        heads, head_size = 12, 64
        decoder_input: dict[str, np.ndarray] = {
            "input_ids": np.array([[0]], dtype=np.int64),
            "encoder_hidden_states": encoder_out,
            "use_cache_branch": np.array([True], dtype=bool),
        }
        for layer in range(self.num_layers):
            for kind in ("key", "value"):
                decoder_input[f"past_key_values.{layer}.{kind}"] = np.zeros(
                    (1, heads, 0, head_size), dtype=np.float32
                )

        token_ids: list[int] = []
        generated = ""
        for _ in range(max_length):
            outputs = self.decoder.run(None, decoder_input)
            token_id = int(np.argmax(outputs[0][0, -1]))
            if token_id == self.tokenizer.eos_token_id:
                break
            token_ids.append(token_id)
            generated = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            if _has_long_repetition(generated):
                break
            decoder_input["input_ids"] = np.array([[token_id]], dtype=np.int64)
            for layer in range(self.num_layers):
                decoder_input[f"past_key_values.{layer}.key"] = outputs[layer * 2 + 1]
                decoder_input[f"past_key_values.{layer}.value"] = outputs[layer * 2 + 2]
        return generated.strip()


def get_model(model_dir: Path) -> MixTeXModel:
    key = str(model_dir.resolve()).casefold()
    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = MixTeXModel(model_dir)
        return _MODEL_CACHE[key]


def _split_unescaped(text: str, separator: str) -> list[str]:
    result: list[str] = []
    current: list[str] = []
    depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif char == "}" and (index == 0 or text[index - 1] != "\\"):
            depth = max(0, depth - 1)
        if depth == 0 and text.startswith(separator, index):
            result.append("".join(current))
            current = []
            index += len(separator)
            continue
        current.append(char)
        index += 1
    result.append("".join(current))
    return result


def _unwrap_commands(text: str) -> str:
    previous = None
    command = re.compile(
        r"\\(?:text|textrm|textbf|mathrm|mathbf|mathit|operatorname|makecell|mbox)\s*\{([^{}]*)\}"
    )
    while previous != text:
        previous = text
        text = command.sub(r"\1", text)
    return text


def latex_cell_to_text(text: str) -> str:
    text = re.sub(r"\\(?:hline|toprule|midrule|bottomrule)\b", "", text)
    text = re.sub(r"\\(?:cline|cmidrule)\s*\{[^{}]*\}", "", text)
    text = _unwrap_commands(text)
    text = text.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
    text = text.replace("\\#", "#").replace("~", " ")
    text = re.sub(r"\\(?:quad|qquad)\b", " ", text)
    text = re.sub(r"\\[,;!:]", " ", text)
    text = re.sub(r"\\(?:newline|linebreak|cr)\b", "\n", text)
    text = text.replace("$", "")
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def parse_latex_tables(latex: str, source: str = "") -> list[tuple[list[list[str]], list[tuple[int, int, int, int]]]]:
    latex = re.sub(r"```(?:latex|tex)?", "", latex, flags=re.I).replace("```", "")
    environments = list(
        re.finditer(
            r"\\begin\{(tabular\*?|array)\}(?:\{[^{}]*\})?(.*?)\\end\{\1\}",
            latex,
            flags=re.I | re.S,
        )
    )
    bodies = [match.group(2) for match in environments]
    if not bodies and "&" in latex and "\\\\" in latex:
        bodies = [latex]

    parsed: list[tuple[list[list[str]], list[tuple[int, int, int, int]]]] = []
    for body in bodies:
        body = re.sub(r"\\(?:hline|toprule|midrule|bottomrule)\b", "", body)
        body = re.sub(r"\\(?:cline|cmidrule)\s*\{[^{}]*\}", "", body)
        rows: list[list[str]] = []
        merges: list[tuple[int, int, int, int]] = []
        for raw_row in _split_unescaped(body, "\\\\"):
            raw_row = raw_row.strip()
            if not raw_row:
                continue
            row: list[str] = []
            for raw_cell in _split_unescaped(raw_row, "&"):
                multicolumn = re.fullmatch(
                    r"\s*\\multicolumn\{(\d+)\}\{[^{}]*\}\{(.*)\}\s*",
                    raw_cell,
                    flags=re.S,
                )
                if multicolumn:
                    span = max(1, int(multicolumn.group(1)))
                    start = len(row)
                    row.append(latex_cell_to_text(multicolumn.group(2)))
                    row.extend([""] * (span - 1))
                    if span > 1:
                        merges.append((len(rows), start, len(rows), start + span - 1))
                else:
                    row.append(latex_cell_to_text(raw_cell))
            if any(cell for cell in row):
                rows.append(row)
        if rows:
            width = max(map(len, rows))
            parsed.append(([row + [""] * (width - len(row)) for row in rows], merges))
    return parsed


def _horizontal_grid_lines(image: np.ndarray) -> list[int]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, image.shape[1] // 5), 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    scores = np.count_nonzero(horizontal, axis=1)
    candidates = np.flatnonzero(scores >= image.shape[1] * 0.35)
    groups: list[list[int]] = []
    for value in candidates:
        if not groups or value > groups[-1][-1] + 1:
            groups.append([int(value)])
        else:
            groups[-1].append(int(value))
    return [round(sum(group) / len(group)) for group in groups]


def _table_bands(image: Image.Image, rows_per_band: int = 6) -> list[Image.Image]:
    def fixed_vertical_bands() -> list[Image.Image]:
        if image.height <= image.width * 1.15:
            return [image]
        band_height = max(240, round(image.width * 0.30))
        overlap = max(20, round(band_height * 0.06))
        result: list[Image.Image] = []
        top = 0
        while top < image.height:
            bottom = min(image.height, top + band_height)
            result.append(image.crop((0, top, image.width, bottom)))
            if bottom == image.height:
                break
            top = bottom - overlap
        return result

    rgb = np.asarray(image.convert("RGB"))
    lines = _horizontal_grid_lines(rgb)
    if len(lines) < rows_per_band + 2:
        # A photographed page may have slightly slanted lines, so the strict
        # horizontal-line detector can miss them. Avoid shrinking a tall page
        # into one 448px input; use overlapping vertical windows instead.
        return fixed_vertical_bands()
    top, bottom = lines[0], lines[-1]
    if bottom - top < image.height * 0.45:
        return fixed_vertical_bands()
    bands: list[Image.Image] = []
    for start in range(0, len(lines) - 1, rows_per_band):
        end = min(start + rows_per_band, len(lines) - 1)
        y1 = max(0, lines[start] - 4)
        y2 = min(image.height, lines[end] + 5)
        if y2 - y1 >= 20:
            bands.append(image.crop((0, y1, image.width, y2)))
    return bands or [image]


def recognize_with_mixtex(
    path: Path,
    model_dir: Path,
    base_module: object,
) -> list[object]:
    rectified = base_module.rectify_table_image(path)
    image = Image.open(io.BytesIO(rectified)).convert("RGB")
    model = get_model(model_dir)
    bands = _table_bands(image)
    all_rows: list[list[str]] = []
    all_merges: list[tuple[int, int, int, int]] = []
    raw_parts: list[str] = []

    for band_number, band in enumerate(bands):
        latex = model.recognize(band)
        raw_parts.append(f"% band {band_number + 1}/{len(bands)}\n{latex}")
        parsed = [
            item
            for item in parse_latex_tables(latex, path.name)
            if len(item[0]) >= 2 and max(map(len, item[0]), default=0) >= 2
        ]
        if not parsed:
            continue
        rows, merges = max(parsed, key=lambda item: len(item[0]) * max(map(len, item[0])))
        if band_number and all_rows and rows:
            normalized_first = [re.sub(r"\s+", "", cell) for cell in rows[0]]
            normalized_header = [re.sub(r"\s+", "", cell) for cell in all_rows[0]]
            if normalized_first == normalized_header:
                rows = rows[1:]
                merges = [
                    (r1 - 1, c1, r2 - 1, c2)
                    for r1, c1, r2, c2 in merges
                    if r1 > 0 and r2 > 0
                ]
        offset = len(all_rows)
        all_rows.extend(rows)
        all_merges.extend((r1 + offset, c1, r2 + offset, c2) for r1, c1, r2, c2 in merges)

    raw_path = path.with_name(f"{path.stem}_mixtex_raw.tex")
    raw_path.write_text("\n\n".join(raw_parts), encoding="utf-8")
    if not all_rows:
        raise RuntimeError(
            f"MixTeX 没有生成可解析的 tabular/array 表格。原始结果已保存到：{raw_path}"
        )
    header_rows = base_module.detect_header_rows(all_rows)
    if header_rows and all_rows:
        all_rows[0] = base_module.normalize_known_header(all_rows[0])
    return [
        base_module.ParsedTable(
            rows=all_rows,
            merges=all_merges,
            header_rows=header_rows,
            source=path.name,
        )
    ]
