from __future__ import annotations

import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk


ROOT = Path(__file__).resolve().parent
REPO = ROOT / "MixTeX-Latex-OCR"
MODEL_DIR = REPO / "onnx"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

import table_recognizer_gui as base  # noqa: E402
from mixtex_table_adapter import recognize_with_mixtex  # noqa: E402


def recognize_image(path: Path, model_path: str, timeout: int = 180) -> list[base.ParsedTable]:
    del timeout
    chosen = Path(model_path.strip()) if model_path.strip() else MODEL_DIR
    return recognize_with_mixtex(path, chosen, base)  # type: ignore[return-value]


def main() -> None:
    base.DEFAULT_API = str(MODEL_DIR)
    base.recognize_image = recognize_image
    root = base.TkinterDnD.Tk() if base.TkinterDnD else tk.Tk()
    try:
        ttk.Style(root).theme_use("vista")
    except tk.TclError:
        pass
    app = base.TableRecognizerApp(root)
    root.title("MixTeX 本地表格识别工具（实验版）")
    app.status.set("拖入图片；模型首次加载约需数秒。适合清晰、较简单的表格。")
    root.mainloop()


if __name__ == "__main__":
    main()

