# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for table_recognizer_gui_paddleocr_vl.

Builds a single-file exe that runs the PaddleOCR-VL-1.6 table-recognition GUI on
a machine without Python.  It talks to the remote vLLM server over HTTP, so no
model weights are bundled.

Usage:
    pyinstaller table_ocr_paddleocr_vl.spec --noconfirm
"""

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

datas = []
binaries = []
hiddenimports = []

# tkinterdnd2 ships Tcl/Tk resources that must be bundled for drag & drop.
_datas, _binaries, _hidden = collect_all("tkinterdnd2")
datas += _datas
binaries += _binaries
hiddenimports += _hidden

# Ensure the rich libraries and their plugins/attrs are fully packaged.
for pkg in ("cv2", "PIL", "numpy", "openpyxl"):
    _d, _b, _h = collect_all(pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

hiddenimports += [
    "table_recognizer_gui",
    "paddle_ocr_vl_adapter",
    "preprocess_frame",
    "image_preprocess",
    "tkinter",
    "tkinter.ttk",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "tkinterdnd2.TkinterDnD",
]

a = Analysis(
    ["table_recognizer_gui_paddleocr_vl.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "onnxruntime",
        "MixTeX",
        "torch",
        "paddle",
        "paddleocr",
        "matplotlib",
    ],
    noarchive=False,
    optimize=0,
)

# --------------------------------------------------------------------------- #
# Trim the bundle: the app only does still-image work (decode/resize/warp), so
# OpenCV's video/ffmpeg backends are dead weight, and we only need one OpenCV
# dist-info.  Drop them to shrink the exe substantially without losing function.
# --------------------------------------------------------------------------- #
def _keep_binary(entry) -> bool:
    # entry = (dest_name, src, typecode)
    dest = str(entry[0]).lower()
    if "videoio" in dest or "ffmpeg" in dest or "gapi" in dest:
        return False
    if "opencv_python" in dest and ".dist-info" in dest:
        return False
    if dest.endswith(".dist-info"):
        return False
    return True


a.binaries = [b for b in a.binaries if _keep_binary(b)]
a.datas = [d for d in a.datas if ".dist-info" not in str(d[0]).lower()]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="表格识别_PaddleOCR_VL",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # GUI app: no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
