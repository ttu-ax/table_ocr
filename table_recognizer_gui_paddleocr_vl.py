from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import table_recognizer_gui as base  # noqa: E402
from paddle_ocr_vl_adapter import DEFAULT_API, recognize_with_paddle_ocr_vl  # noqa: E402


def recognize_image(path: Path, api_url: str, timeout: int = 300) -> list[base.ParsedTable]:
    chosen = api_url.strip() if api_url.strip() else DEFAULT_API
    return recognize_with_paddle_ocr_vl(path, chosen, timeout)  # type: ignore[return-value]


def main() -> None:
    base.DEFAULT_API = DEFAULT_API
    base.recognize_image = recognize_image  # type: ignore[assignment]
    root = base.TkinterDnD.Tk() if base.TkinterDnD else __import__("tkinter").Tk()
    try:
        __import__("tkinter.ttk", fromlist=["Style"]).Style(root).theme_use("vista")
    except Exception:
        pass
    app = base.TableRecognizerApp(root)
    root.title("PaddleOCR-VL-1.6 表格识别工具（vLLM 远程）")
    app.status.set("拖入图片；调用 192.168.10.26:8003 的 PaddleOCR-VL-1.6 模型。")
    root.mainloop()


if __name__ == "__main__":
    main()
