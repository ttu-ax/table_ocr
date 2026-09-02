"""PaddleOCR-VL-1.6 表格识别效果测试脚本（命令行）。

用法示例：
    python test_paddle_ocr_vl.py " (13).jpg" " (14).jpg"
    python test_paddle_ocr_vl.py --dir ./some_dir
    python test_paddle_ocr_vl.py --api http://192.168.10.26:8003/v1/chat/completions " (15).jpg"

对每张图片：
  1. 透视矫正（复用 table_recognizer_gui.rectify_table_image）
  2. 限制分辨率到模型像素/令牌预算内
  3. 调用 vLLM 的 PaddleOCR-VL-1.6，提示词 "Table Recognition:"
  4. 把 OTSL 输出转成 HTML，再复用 base 的 HTML 表格解析 -> ParsedTable
  5. 打印表格行列数、表头、前几行与底部汇总行
  6. 将原始 OTSL 与 HTML 保存到 *_otsl.txt / *_html.txt 便于核对模型输出
  7. 全部结果写入一个 XLSX（含表头着色、冻结窗格、合并单元格）
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import table_recognizer_gui as base
from paddle_ocr_vl_adapter import DEFAULT_API, recognize_with_paddle_ocr_vl_debug


def _summarize(tables) -> str:
    parts = []
    for index, table in enumerate(tables):
        width = max((len(row) for row in table.rows), default=0)
        parts.append(
            f"表{index}: {len(table.rows)}行 x {width}列, 表头行={table.header_rows}, "
            f"合并单元格={len(table.merges)}"
        )
    return " | ".join(parts)


def _echo_row(row: list[str], limit: int = 12) -> list[str]:
    return [str(cell)[:limit] for cell in row]


def run_one(path: Path, api_url: str, timeout: int, persist: bool):
    started = time.perf_counter()
    tables, otsl, html_str = recognize_with_paddle_ocr_vl_debug(path, api_url, timeout)
    elapsed = time.perf_counter() - started
    print(f"[{path.name}] {elapsed:.1f}s  {_summarize(tables)}")
    for index, table in enumerate(tables):
        print(f"    表{index} 表头/首行:")
        for row in table.rows[:2]:
            print("      ", _echo_row(row))
        if len(table.rows) > 3:
            print("      ...")
            print("      末行:", _echo_row(table.rows[-1]))
    if persist:
        path.with_name(f"{path.stem}_otsl.txt").write_text(otsl, encoding="utf-8")
        path.with_name(f"{path.stem}_html.txt").write_text(html_str, encoding="utf-8")
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description="PaddleOCR-VL-1.6 表格识别测试")
    parser.add_argument("images", nargs="*", help="图片路径")
    parser.add_argument("--dir", help="从目录批量读取图片")
    parser.add_argument("--api", default=DEFAULT_API, help="vLLM chat/completions 地址")
    parser.add_argument("--save", action="store_true", help="同时保存 *_otsl.txt / *_html.txt")
    parser.add_argument("--timeout", type=int, default=300, help="单图超时（秒）")
    parser.add_argument("--out", default="paddleocr_vl_test.xlsx", help="输出 XLSX 文件名")
    args = parser.parse_args()

    paths: list[Path] = []
    if args.dir:
        paths.extend(
            sorted(
                p for p in Path(args.dir).iterdir() if p.suffix.casefold() in base.IMAGE_SUFFIXES
            )
        )
    for name in args.images:
        paths.append(Path(name))
    if not paths:
        print("未提供图片。用法示例：python test_paddle_ocr_vl.py \" (13).jpg\"")
        return

    groups: list[base.TableGroup] = []
    for path in paths:
        if not path.exists():
            print(f"[跳过] 找不到 {path}")
            continue
        try:
            groups.append(base.TableGroup(title=path.stem, tables=run_one(path, args.api, args.timeout, args.save)))
        except Exception as exc:  # noqa: BLE001
            print(f"[{path.name}] 失败：{type(exc).__name__}: {exc}")

    if groups:
        output = base.next_available_path(Path(args.out))
        base.write_groups_to_xlsx(groups, output)
        print(f"\n已生成：{output}")


if __name__ == "__main__":
    main()
