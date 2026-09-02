"""Preprocessing tab: four-point perspective correction + eyedropper brush painter.

Embedded as the "图片预处理" notebook tab inside ``TableRecognizerApp``.  The
frame works on the *original* (un-rectified) image; the user drags four corner
points to define the document quad and paints over hand-written marks / grid
lines / noise.  The result is a JPEG that the recognition pipeline consumes in
place of the raw photo, which improves OCR accuracy.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import image_preprocess as ip


class PreprocessFrame(ttk.Frame):
    """Four-point perspective correction + eyedropper brush overlay painter."""

    UNDO_LIMIT = 20

    # --- tool ids ---------------------------------------------------------- #
    TOOL_DRAG = "drag"
    TOOL_MANUAL = "manual"
    TOOL_EYEDROPPER = "eyedropper"
    TOOL_BRUSH = "brush"

    def __init__(self, master, files: list[Path], app) -> None:
        super().__init__(master)
        self.files = files          # shared reference to the app's file list
        self.app = app              # TableRecognizerApp (for callbacks / log / events)
        self._original: np.ndarray | None = None     # BGR original (un-warped)
        self._quad: ip.Quad | None = None
        self._painter = ip.Painter()
        self._undo: list[ip.Stroke] = []
        self._redo: list[ip.Stroke] = []
        self._tool = self.TOOL_DRAG
        self._color_bgr: tuple[int, int, int] = (0, 0, 0)  # BGR, default black
        self._brush_radius = tk.IntVar(value=12)
        self._add_white_border = tk.BooleanVar(value=False)
        self._use_rectify = tk.BooleanVar(value=True)
        self._apply_paint = tk.BooleanVar(value=True)
        self._preview_warped: np.ndarray | None = None

        # Canvas view transform: image pixel -> canvas coords
        self._view_zoom = 1.0
        self._view_ox = 0.0     # canvas dx
        self._view_oy = 0.0     # canvas dy
        self._drag_corner: int | None = None
        self._manual_points: list[tuple[float, float]] = []
        self._active_stroke: ip.Stroke | None = None
        self._cached_photo: ImageTk.PhotoImage | None = None
        self._cached_original_id: int | None = None
        self._mouse_canvas = (0.0, 0.0)

        # Per-image pending edits, keyed by the resolved path.  Each entry is
        # {"quad": Quad|None, "strokes": [Stroke, ...]} so switching images keeps
        # each image's own four-corner + paint work, and "保存全部改动" can flush
        # every edited image to the OCR cache in one go.
        self._edits: dict[str, dict] = {}

        self._build_ui()
        self._current_index = -1
        self._bind_events()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # --- top bar: navigation + status ---------------------------------- #
        nav = ttk.Frame(self, padding=(8, 6, 8, 2))
        nav.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Button(nav, text="◀ 上一张", command=lambda: self.switch(-1)).pack(side="left")
        ttk.Button(nav, text="下一张 ▶", command=lambda: self.switch(1)).pack(side="left", padx=6)
        ttk.Label(nav, textvariable=tk.StringVar(master=self, value="")).pack(side="left")
        self._title_var = tk.StringVar(value="未选择图片")
        ttk.Label(nav, text="当前图片：").pack(side="left", padx=(16, 0))
        ttk.Label(nav, textvariable=self._title_var).pack(side="left")
        ttk.Button(nav, text="加载当前选中", command=self.load_current_selection).pack(side="right")

        # --- canvas + scrollbars ------------------------------------------- #
        canvas_frame = ttk.Frame(self, padding=8)
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_frame, bg="#3c3c3c", highlightthickness=0)
        self.hbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self._scroll_x)
        self.vbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self._scroll_y)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.canvas.bind("<Configure>", lambda _e: (self._clamp_view(), self._render()))

        # --- right toolbar (scrollable) -------------------------------------- #
        toolbar_wrap = ttk.Frame(self, padding=(0, 8, 8, 8))
        toolbar_wrap.grid(row=1, column=1, sticky="ns")
        toolbar_wrap.rowconfigure(0, weight=1)
        toolbar_wrap.columnconfigure(0, weight=1)
        self.toolbar_canvas = tk.Canvas(toolbar_wrap, highlightthickness=0, width=230)
        self.toolbar_vsb = ttk.Scrollbar(toolbar_wrap, orient="vertical",
                                         command=self.toolbar_canvas.yview)
        self.toolbar_canvas.configure(yscrollcommand=self.toolbar_vsb.set)
        self.toolbar_canvas.grid(row=0, column=0, sticky="nsew")
        self.toolbar_vsb.grid(row=0, column=1, sticky="ns")
        bar = ttk.Frame(self.toolbar_canvas, padding=8)
        self._toolbar_inner = bar
        self._toolbar_window = self.toolbar_canvas.create_window((0, 0), window=bar, anchor="nw")
        bar.bind("<Configure>", self._on_toolbar_size)
        self.toolbar_canvas.bind("<Configure>", self._on_toolbar_view)
        self.toolbar_canvas.bind("<MouseWheel>", lambda e: self.toolbar_canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))

        bar.columnconfigure(0, weight=1)

        # Perspective group
        pg = ttk.LabelFrame(bar, text="透视矫正", padding=8)
        pg.pack(fill="x", pady=(0, 8))
        ttk.Button(pg, text="自动四点矫正", command=self.auto_rectify).pack(fill="x")
        self._tool_var = tk.StringVar(master=self, value=self.TOOL_DRAG)
        ttk.Radiobutton(pg, text="拖动角点", value=self.TOOL_DRAG,
                        variable=self._tool_var, command=self._set_tool).pack(fill="x")
        ttk.Radiobutton(pg, text="手动取四点", value=self.TOOL_MANUAL,
                        variable=self._tool_var, command=self.start_manual_points).pack(fill="x")
        ttk.Button(pg, text="重置角点", command=self.reset_corners).pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(pg, text="矫正后加白边", variable=self._add_white_border).pack(fill="x", anchor="w")

        # Color group
        cg = ttk.LabelFrame(bar, text="取色涂抹", padding=8)
        cg.pack(fill="x", pady=(0, 8))
        ttk.Radiobutton(cg, text="吸管取色", value=self.TOOL_EYEDROPPER,
                        variable=self._tool_var, command=self._set_tool).pack(anchor="w")
        ttk.Radiobutton(cg, text="画笔涂抹", value=self.TOOL_BRUSH,
                        variable=self._tool_var, command=self._set_tool).pack(anchor="w")
        ttk.Label(cg, text="笔刷大小").pack(anchor="w", pady=(6, 0))
        ttk.Scale(cg, from_=3, to=60, variable=self._brush_radius).pack(fill="x")
        # color swatch + value
        swatch_row = ttk.Frame(cg)
        swatch_row.pack(fill="x", pady=(6, 0))
        self._swatch = tk.Label(swatch_row, text="  ", bg="#000000", width=4)
        self._swatch.pack(side="left")
        self._color_label = ttk.Label(swatch_row, text="RGB(0, 0, 0)")
        self._color_label.pack(side="left", padx=6)
        self._palette = TkHsvPalette(cg, on_color=self._on_palette_color)
        self._palette.pack(fill="x", pady=(6, 0))

        # Undo group
        ug = ttk.LabelFrame(bar, text="操作", padding=8)
        ug.pack(fill="x", pady=(0, 8))
        ttk.Button(ug, text="撤销", command=self.undo).pack(fill="x")
        ttk.Button(ug, text="重做", command=self.redo).pack(fill="x", pady=(4, 0))
        ttk.Button(ug, text="清空涂抹", command=self.clear_paint).pack(fill="x", pady=(4, 0))

        # Apply group
        ag = ttk.LabelFrame(bar, text="应用", padding=8)
        ag.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(ag, text="透视矫正", variable=self._use_rectify).pack(anchor="w")
        ttk.Checkbutton(ag, text="涂抹覆盖", variable=self._apply_paint).pack(anchor="w")
        ttk.Button(ag, text="✍ 保存全部改动", command=self.save_all_changes).pack(fill="x", pady=(6, 2))
        ttk.Button(ag, text="应用到当前图", command=self.apply_to_current).pack(fill="x", pady=2)
        ttk.Button(ag, text="套用当前参数到全部", command=self.apply_to_all).pack(fill="x", pady=2)
        ttk.Button(ag, text="预览矫正结果", command=self.preview_warp).pack(fill="x", pady=2)
        ttk.Button(ag, text="重置本图", command=self.reset_image).pack(fill="x", pady=(2, 2))
        ttk.Button(ag, text="重置全部改动", command=self.reset_all).pack(fill="x", pady=2)

    # ------------------------------------------------------------------ #
    # Event bindings
    # ------------------------------------------------------------------ #
    def _bind_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._on_wheel(e, up=True))
        self.canvas.bind("<Button-5>", lambda e: self._on_wheel(e, up=False))
        self.canvas.bind("<Motion>", self._on_hover)

    def _set_tool(self) -> None:
        self._tool = self._tool_var.get()

    # --- right-toolbar scroll sizing ---------------------------------------- #
    def _on_toolbar_size(self, _event) -> None:
        self.toolbar_canvas.configure(scrollregion=self.toolbar_canvas.bbox("all"))

    def _on_toolbar_view(self, _event) -> None:
        # Match the inner frame width to the canvas width so buttons stretch.
        self.toolbar_canvas.itemconfigure(self._toolbar_window, width=self.toolbar_canvas.winfo_width())

    # ------------------------------------------------------------------ #
    # Canvas -> image coordinate mapping
    # ------------------------------------------------------------------ #
    def _img_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return (x * self._view_zoom + self._view_ox, y * self._view_zoom + self._view_oy)

    def _canvas_to_img(self, x: float, y: float) -> tuple[float, float]:
        if self._view_zoom <= 0:
            return (0.0, 0.0)
        return ((x - self._view_ox) / self._view_zoom, (y - self._view_oy) / self._view_zoom)

    # ------------------------------------------------------------------ #
    # Navigation / load
    # ------------------------------------------------------------------ #
    def files_changed(self) -> None:
        """Called by the app when the shared file list changes."""
        if self._current_index >= len(self.files):
            self._current_index = len(self.files) - 1
        if self._current_index < 0:
            self._original = None
            self._quad = None
            self._title_var.set("未选择图片")
            self._render()
            return
        # Re-load only if the file at the current index actually differs.
        if 0 <= self._current_index < len(self.files):
            self._load_current()

    def switch(self, delta: int) -> None:
        if not self.files:
            return
        if self._current_index < 0:
            self._current_index = 0
        self._current_index = max(0, min(len(self.files) - 1, self._current_index + delta))
        self._load_current()

    def load_current_selection(self) -> None:
        if not self.files:
            return
        sel = self.app.listbox.curselection() if hasattr(self.app, "listbox") else ()
        if sel:
            self._current_index = sel[0]
        elif self._current_index < 0:
            self._current_index = 0
        self._load_current()

    # --- per-image edit state --------------------------------------------- #
    def _current_key(self) -> str:
        if 0 <= self._current_index < len(self.files):
            return str(Path(self.files[self._current_index]).resolve()).casefold()
        return ""

    def _snapshot_current(self) -> None:
        """Record the current image's quad + strokes so it survives switching."""
        key = self._current_key()
        if not key:
            return
        self._edits[key] = {
            "quad": self._quad,
            "strokes": list(self._painter.strokes),
        }

    def is_modified(self, path: Path) -> bool:
        """Whether this file has pending (unsaved) preprocessing edits."""
        return str(Path(path).resolve()).casefold() in self._edits

    def _mark_modified(self) -> None:
        """Snapshot the current edit and refresh the list's ●已修改 marker."""
        self._snapshot_current()
        self._refresh_marker()

    def _refresh_marker(self) -> None:
        self.app.refresh_preprocess_markers()

    def _load_current(self) -> None:
        if not (0 <= self._current_index < len(self.files)):
            self._title_var.set("未选择图片")
            self._original = None
            self._quad = None
            self._painter.clear()
            self._undo.clear()
            self._redo.clear()
            self._render()
            return
        path = self.files[self._current_index]
        try:
            encoded = np.fromfile(str(path), dtype=np.uint8)
            self._original = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        except Exception:
            self._original = None
        if self._original is None:
            self._title_var.set(f"无法读取：{path.name}")
            return
        self._title_var.set(f"{self._current_index + 1}/{len(self.files)} {path.name}")
        # Restore this image's own saved edits (quad + strokes), falling back to a
        # fresh auto-detected quad.
        key = self._current_key()
        saved = self._edits.get(key)
        if saved is not None:
            self._quad = saved.get("quad")
            self._painter = ip.Painter()
            for stroke in saved.get("strokes", []):
                self._painter.add_stroke(stroke)
        else:
            self._painter = ip.Painter()
            self.auto_rectify()
        self._undo.clear()
        self._redo.clear()
        self._fit_zoom()
        self._render()

    def _fit_zoom(self) -> None:
        if self._original is None:
            self._view_zoom = 1.0
            return
        w = self.canvas.winfo_width() or 1
        h = self.canvas.winfo_height() or 1
        ih, iw = self._original.shape[:2]
        self._view_zoom = max(0.02, min(w / iw, h / ih))
        # Center the image; when larger than the viewport this becomes negative
        # (the scrollbars then let the user pan).
        self._view_ox = (w - iw * self._view_zoom) / 2
        self._view_oy = (h - ih * self._view_zoom) / 2

    # ------------------------------------------------------------------ #
    # Scrollbars (pan) + view clamping
    # ------------------------------------------------------------------ #
    def _scroll_x(self, *args: str) -> None:
        if self._original is None:
            return
        # args: ("moveto", frac) or ("scroll", units, "units"|"pages")
        w = self.canvas.winfo_width() or 1
        iw = self._original.shape[1]
        img_w = iw * self._view_zoom
        if img_w <= w:
            self._view_ox = (w - img_w) / 2
        elif args[0] == "moveto":
            self._view_ox = -float(args[1]) * (img_w - w)
        elif args[0] == "scroll":
            units, what = int(args[1]), args[2]
            step = w * 0.9 if what == "pages" else (40 if what == "units" else 40)
            self._view_ox -= units * step
        self._clamp_view_x(img_w, w)
        self._render_pan()

    def _scroll_y(self, *args: str) -> None:
        if self._original is None:
            return
        h = self.canvas.winfo_height() or 1
        ih = self._original.shape[0]
        img_h = ih * self._view_zoom
        if img_h <= h:
            self._view_oy = (h - img_h) / 2
        elif args[0] == "moveto":
            self._view_oy = -float(args[1]) * (img_h - h)
        elif args[0] == "scroll":
            units, what = int(args[1]), args[2]
            step = h * 0.9 if what == "pages" else (40 if what == "units" else 40)
            self._view_oy -= units * step
        self._clamp_view_y(img_h, h)
        self._render_pan()

    def _clamp_view(self) -> None:
        if self._original is None:
            return
        w = self.canvas.winfo_width() or 1
        h = self.canvas.winfo_height() or 1
        self._clamp_view_x(self._original.shape[1] * self._view_zoom, w)
        self._clamp_view_y(self._original.shape[0] * self._view_zoom, h)

    def _clamp_view_x(self, img_w: float, view_w: float) -> None:
        if img_w <= view_w:
            self._view_ox = (view_w - img_w) / 2
        else:
            self._view_ox = min(0.0, max(view_w - img_w, self._view_ox))

    def _clamp_view_y(self, img_h: float, view_h: float) -> None:
        if img_h <= view_h:
            self._view_oy = (view_h - img_h) / 2
        else:
            self._view_oy = min(0.0, max(view_h - img_h, self._view_oy))

    # ------------------------------------------------------------------ #
    # Four-point correction
    # ------------------------------------------------------------------ #
    def auto_rectify(self) -> None:
        if self._original is None:
            return
        quad = ip.detect_document_quad(self._original)
        self._quad = quad
        self._render()

    def reset_corners(self) -> None:
        self.auto_rectify()

    # --- Manual four-point placement ---------------------------------------- #
    def start_manual_points(self) -> None:
        """Enter manual mode: clicking four corners builds the quad."""
        self._tool = self.TOOL_MANUAL
        self._manual_points = []
        self._quad = None
        self._render()

    def _finish_manual_points(self) -> None:
        if len(self._manual_points) == 4:
            # Click order does not matter: sort the four points into TL/TR/BR/BL
            # so the perspective warp is correct regardless of click order.
            ordered = ip.order_quad(np.array(self._manual_points, dtype=np.float32))
            self._quad = ip.Quad(ordered)
        self._manual_points = []
        if self._quad is not None:
            self._tool = self.TOOL_DRAG
            self._tool_var.set(self.TOOL_DRAG)
            self._mark_modified()
        self._render()

    def _add_manual_point(self, img_pt: tuple[float, float]) -> None:
        self._manual_points.append(self._clamp_to_image(img_pt))
        if len(self._manual_points) >= 4:
            self._finish_manual_points()
        else:
            self._render()

    def _clamp_to_image(self, pt: tuple[float, float]) -> tuple[float, float]:
        if self._original is None:
            return pt
        h, w = self._original.shape[:2]
        return (float(np.clip(pt[0], 0, w - 1)), float(np.clip(pt[1], 0, h - 1)))

    def _set_corner(self, index: int, img_pt: tuple[float, float]) -> None:
        if self._quad is None:
            return
        pts = self._quad.points.tolist()
        pts[index] = list(self._clamp_to_image(img_pt))
        self._quad = ip.Quad(np.array(pts, dtype=np.float32).reshape(4, 2))

    def _hit_corner(self, img_pt: tuple[float, float], tol_px: float = 16.0) -> int | None:
        if self._quad is None:
            return None
        x, y = img_pt
        for i, (qx, qy) in enumerate(self._quad.points.tolist()):
            if abs(qx - x) * self._view_zoom <= tol_px and abs(qy - y) * self._view_zoom <= tol_px:
                return i
        return None

    # ------------------------------------------------------------------ #
    # Mouse handlers
    # ------------------------------------------------------------------ #
    def _on_wheel(self, event, up: bool | None = None) -> None:
        if self._original is None:
            return
        delta = 1.1 if (up or int(getattr(event, "delta", 0)) > 0) else 0.9
        mx, my = self._mouse_canvas
        img_x, img_y = self._canvas_to_img(mx, my)
        self._view_zoom = max(0.02, min(8.0, self._view_zoom * delta))
        self._view_ox = mx - img_x * self._view_zoom
        self._view_oy = my - img_y * self._view_zoom
        self._clamp_view()
        self._render()

    def _on_hover(self, event) -> None:
        self._mouse_canvas = (float(event.x), float(event.y))
        self._update_cursor()

    def _update_cursor(self) -> None:
        if self._original is None:
            self.canvas.configure(cursor="arrow")
            return
        img_pt = self._canvas_to_img(*self._mouse_canvas)
        if self._tool == self.TOOL_BRUSH:
            self.canvas.configure(cursor="circle")
        elif self._tool == self.TOOL_EYEDROPPER:
            self.canvas.configure(cursor="crosshair")
        elif self._tool == self.TOOL_MANUAL:
            self.canvas.configure(cursor="crosshair")
        elif self._hit_corner(img_pt) is not None:
            self.canvas.configure(cursor="fleur")
        else:
            self.canvas.configure(cursor="arrow")

    def _on_press(self, event) -> None:
        if self._original is None:
            return
        img_pt = self._canvas_to_img(float(event.x), float(event.y))
        if self._tool == self.TOOL_EYEDROPPER:
            self._sample_color(img_pt)
            return
        if self._tool == self.TOOL_MANUAL:
            self._add_manual_point(img_pt)
            return
        if self._tool == self.TOOL_BRUSH:
            self._active_stroke = ip.Stroke(color=self._color_bgr, radius=int(self._brush_radius.get()))
            self._active_stroke.points.append(img_pt)
            self._render_overlay()
            return
        # DRAG tool: grab a corner if near one
        hit = self._hit_corner(img_pt)
        if hit is not None:
            self._drag_corner = hit
        else:
            self.app.log_status("未检测到自动四点，请点「手动取四点」在图上点选 4 个角点")

    def _on_motion(self, event) -> None:
        if self._original is None:
            return
        img_pt = self._canvas_to_img(float(event.x), float(event.y))
        if self._tool == self.TOOL_BRUSH and self._active_stroke is not None:
            last = self._active_stroke.points[-1]
            if abs(img_pt[0] - last[0]) + abs(img_pt[1] - last[1]) >= 1.0:
                self._active_stroke.points.append(img_pt)
            self._render_overlay()
            return
        if self._drag_corner is not None and self._quad is not None:
            self._set_corner(self._drag_corner, img_pt)
            self._render_overlay()

    def _on_release(self, event) -> None:
        if self._tool == self.TOOL_BRUSH and self._active_stroke is not None:
            if len(self._active_stroke.points) >= 2:
                self._painter.add_stroke(self._active_stroke)
                self._undo.append(self._active_stroke)
                self._undo = self._undo[-self.UNDO_LIMIT:]
                self._redo.clear()
                self._mark_modified()
            self._active_stroke = None
            self._render_overlay()
            return
        if self._drag_corner is not None:
            self._drag_corner = None
            self._mark_modified()

    # ------------------------------------------------------------------ #
    # Color sampling + palette
    # ------------------------------------------------------------------ #
    def _sample_color(self, img_pt: tuple[float, float]) -> None:
        if self._original is None:
            return
        x, y = int(round(img_pt[0])), int(round(img_pt[1]))
        h, w = self._original.shape[:2]
        x, y = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
        b, g, r = (int(v) for v in self._original[y, x])
        self._color_bgr = (b, g, r)
        self._update_swatch()

    def _on_palette_color(self, bgr: tuple[int, int, int]) -> None:
        self._color_bgr = bgr
        self._update_swatch()

    def _update_swatch(self) -> None:
        b, g, r = self._color_bgr
        self._swatch.configure(bg=f"#{r:02x}{g:02x}{b:02x}")
        self._color_label.configure(text=ip.rgb_str(self._color_bgr))

    # ------------------------------------------------------------------ #
    # Undo / redo / clear
    # ------------------------------------------------------------------ #
    def undo(self) -> None:
        stroke = self._painter.pop()
        if stroke is not None:
            self._redo.append(stroke)
            self._render()

    def redo(self) -> None:
        if self._redo:
            stroke = self._redo.pop()
            self._painter.add_stroke(stroke)
            self._undo.append(stroke)
            self._render()

    def clear_paint(self) -> None:
        if self._painter.empty:
            return
        self._painter.clear()
        self._undo.clear()
        self._redo.clear()
        self._render()

    # ------------------------------------------------------------------ #
    # Apply / preview / reset
    # ------------------------------------------------------------------ #
    def _build_processed(self) -> np.ndarray | None:
        """Compose paint (on original) then warp, as the pipeline will consume it."""
        if self._original is None:
            return None
        base_img = self._original
        if self._apply_paint.get() and not self._painter.empty:
            base_img = self._painter.composite(self._original)
        result = base_img
        if self._use_rectify.get() and self._quad is not None:
            result = ip.warp_document(base_img, self._quad, self._add_white_border.get())
        return result

    def apply_to_current(self) -> None:
        result = self._build_processed()
        if result is None:
            return
        if not (0 <= self._current_index < len(self.files)):
            return
        path = self.files[self._current_index]
        self.app.set_preprocessed(path, cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
        self._snapshot_current()
        self._refresh_marker()
        self.app.log_status(f"已应用预处理到 {path.name}")

    def apply_to_all(self) -> None:
        """Apply the *current* image's quad + paint settings to every file."""
        if not self.files or self._original is None:
            return
        current = self._build_processed()
        if current is None:
            return
        for path in list(self.files):
            encoded = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img_out = img
            if self._apply_paint.get() and not self._painter.empty:
                img_out = self._painter.composite(img)
            if self._use_rectify.get() and self._quad is not None:
                img_out = ip.warp_document(img_out, self._quad, self._add_white_border.get())
            self.app.set_preprocessed(path, cv2.imencode(".jpg", img_out, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
            # Record each file as edited so the ●已修改 marker stays consistent.
            key = str(Path(path).resolve()).casefold()
            self._edits.setdefault(key, {})["quad"] = self._quad
            self._edits.setdefault(key, {})["strokes"] = list(self._painter.strokes)
        self._refresh_marker()
        self.app.log_status(f"已将预处理参数应用到全部 {len(self.files)} 张图片")

    def save_all_changes(self) -> None:
        """One-click save: flush every edited image to the OCR cache at once."""
        if not self.files:
            return
        saved = 0
        for path in list(self.files):
            key = str(Path(path).resolve()).casefold()
            edit = self._edits.get(key)
            if edit is None:
                continue
            encoded = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img_out = img
            quad = edit.get("quad")
            strokes = list(edit.get("strokes", []))
            if self._apply_paint.get() and strokes:
                p = ip.Painter()
                for stroke in strokes:
                    p.add_stroke(stroke)
                img_out = p.composite(img)
            if self._use_rectify.get() and quad is not None:
                img_out = ip.warp_document(img_out, quad, self._add_white_border.get())
            self.app.set_preprocessed(path, cv2.imencode(".jpg", img_out, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
            saved += 1
        self._refresh_marker()
        self.app.log_status(f"已保存 {saved} 张图片的预处理改动")

    def reset_image(self) -> None:
        if not (0 <= self._current_index < len(self.files)):
            return
        path = self.files[self._current_index]
        key = str(Path(path).resolve()).casefold()
        self._edits.pop(key, None)
        self.app.clear_preprocessed(path)
        self._painter.clear()
        self._undo.clear()
        self._redo.clear()
        self.auto_rectify()
        self._refresh_marker()
        self.app.log_status(f"已重置 {path.name} 的预处理")

    def reset_all(self) -> None:
        """Clear every edited image's cache and pending edits."""
        self._edits.clear()
        for path in list(self.files):
            self.app.clear_preprocessed(path)
        if self._original is not None:
            self._painter.clear()
            self._undo.clear()
            self._redo.clear()
            self.auto_rectify()
        self._refresh_marker()
        self.app.log_status("已重置全部图片的预处理改动")

    def preview_warp(self) -> None:
        result = self._build_processed()
        if result is None:
            return
        self._preview_warped = result
        self._render_preview()

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render_preview(self) -> None:
        if self._preview_warped is None:
            return
        self._render_base(self._preview_warped, source="preview")

    def _render(self) -> None:
        if self._original is None:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2,
                text="请先在“待识别列表”选择并加载图片", fill="#cccccc",
            )
            return
        self._render_base(self._original, source="original")

    def _apply_view(self, pil_img: Image.Image) -> Image.Image:
        if self._view_zoom <= 0:
            return pil_img
        new_w = max(1, int(pil_img.width * self._view_zoom))
        new_h = max(1, int(pil_img.height * self._view_zoom))
        return pil_img.resize((new_w, new_h), Image.LANCZOS)

    # --- Split "heavy base image" from "light overlay" so drag/paint stays --- #
    # smooth.  The base (a large BGR photo) is converted/resized/encoded into a
    # PhotoImage ONLY when the source or the zoom actually changes; it is cached.
    # Mouse-move actions (corner drag, paint strokes) only redraw the overlay
    # canvas items, which is orders of magnitude cheaper than re-encoding the
    # full image every frame.
    def _on_zoom_changed(self) -> bool:
        """Force a base re-render when the display zoom changes."""
        old = getattr(self, "_base_zoom", None)
        source = getattr(self, "_base_source", None)
        changed = old != self._view_zoom or source != id(self._original)
        self._base_zoom = self._view_zoom
        self._base_source = id(self._original)
        return changed

    def _render_base(self, bgr: np.ndarray, source: str) -> None:
        """Render the underlying image; skip the expensive re-encode if unchanged."""
        self._clamp_view()
        changed = self._on_zoom_changed()
        if changed or source != getattr(self, "_base_display_source", None):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_img = self._apply_view(Image.fromarray(rgb))
            self._cached_photo = ImageTk.PhotoImage(pil_img)
            self._base_display_source = source
        # Always place the photo (cheap) then the overlay.
        self.canvas.delete("all")
        self.canvas.create_image(self._view_ox, self._view_oy, image=self._cached_photo,
                                 anchor="nw", tags=("photo",))
        if source == "original":
            self._render_overlay()
        self._sync_scrollbars()

    def _render_pan(self) -> None:
        """Reposition the (cached) base photo and redraw overlays only."""
        items = self.canvas.find_withtag("photo")
        if items and self._cached_photo is not None:
            self.canvas.coords(items[0], self._view_ox, self._view_oy)
        elif self._cached_photo is not None:
            self.canvas.delete("all")
            self.canvas.create_image(self._view_ox, self._view_oy, image=self._cached_photo,
                                     anchor="nw", tags=("photo",))
        self._render_overlay()
        self._sync_scrollbars()

    def _render_overlay(self) -> None:
        """Redraw only the overlay items (paint / quad / manual / hints)."""
        self.canvas.delete("paint", "manual", "quad", "corner", "hint")
        # Paint strokes, converted to canvas coordinates.
        for stroke in self._painter.strokes + ([self._active_stroke] if self._active_stroke else []):
            if not stroke.points:
                continue
            pts = [self._img_to_canvas(x, y) for x, y in stroke.points]
            r = stroke.radius * self._view_zoom
            hexc = "#%02x%02x%02x" % (stroke.color[2], stroke.color[1], stroke.color[0])
            self.canvas.create_line(*[c for pt in pts for c in pt], fill=hexc,
                                    width=max(1, int(r * 2)), capstyle=tk.ROUND,
                                    joinstyle=tk.ROUND, tags=("paint",))
            for cx, cy in pts:
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=hexc,
                                        outline="", tags=("paint",))

        # Manual four-point placement: render already-placed points and guides.
        if self._tool == self.TOOL_MANUAL and self._manual_points:
            pts = [self._img_to_canvas(x, y) for x, y in self._manual_points]
            for cx, cy in pts:
                self.canvas.create_oval(cx - 7, cy - 7, cx + 7, cy + 7,
                                        fill="#ff9900", outline="#ffffff", width=1,
                                        tags=("manual",))
            if len(pts) >= 2:
                self.canvas.create_line(*[c for pt in pts for c in pt], fill="#ff9900",
                                        width=2, dash=(4, 3), tags=("manual",))
            self.canvas.create_text(
                self.canvas.winfo_width() / 2, 24,
                text=f"手动取四点：已选 {len(self._manual_points)}/4，请在图上点击表格四角",
                fill="#ff9900", font=("Microsoft YaHei UI", 11, "bold"), tags=("manual",),
            )

        # Document quad + draggable corner handles.
        if self._quad is not None:
            pts = [self._img_to_canvas(qx, qy) for qx, qy in self._quad.points.tolist()]
            self.canvas.create_polygon(pts, outline="#00ff00", fill="", width=2,
                                       tags=("quad",))
            for i, (cx, cy) in enumerate(pts):
                self.canvas.create_oval(cx - 6, cy - 6, cx + 6, cy + 6,
                                        fill="#00dd00", outline="#ffffff", width=1,
                                        tags=("corner", f"corner_{i}"))
                self.canvas.create_text(cx, cy - 14, text=str(i + 1), fill="#00ff00",
                                        font=("Arial", 10, "bold"), tags=("corner",))
        elif self._tool != self.TOOL_MANUAL:
            # Auto-detection failed: give the user a clear action hint.
            self.canvas.create_text(
                self.canvas.winfo_width() / 2, self.canvas.winfo_height() - 28,
                text="未自动识别到四点，请在右侧点「手动取四点」在图上点击表格四角",
                fill="#ff5555", font=("Microsoft YaHei UI", 11, "bold"), tags=("hint",),
            )

    def _render_image(self, bgr: np.ndarray, source: str) -> None:
        # Back-compat alias: full render (base + overlay).
        self._render_base(bgr, source)

    def _sync_scrollbars(self) -> None:
        """Refresh the scrollbar thumbs from the current pan position."""
        if self._original is None:
            self.hbar.set(0, 1)
            self.vbar.set(0, 1)
            return
        w = self.canvas.winfo_width() or 1
        h = self.canvas.winfo_height() or 1
        img_w = self._original.shape[1] * self._view_zoom
        img_h = self._original.shape[0] * self._view_zoom
        if img_w <= w:
            self.hbar.set(0, 1)
        else:
            first = -self._view_ox / (img_w - w)
            self.hbar.set(max(0.0, min(1.0, first)),
                          max(0.0, min(1.0, first + w / (img_w - w))))
        if img_h <= h:
            self.vbar.set(0, 1)
        else:
            first = -self._view_oy / (img_h - h)
            self.vbar.set(max(0.0, min(1.0, first)),
                          max(0.0, min(1.0, first + h / (img_h - h))))


class TkHsvPalette(ttk.Frame):
    """A compact self-drawn HSV color picker.

    Hue runs left-to-right and value top-to-bottom; a saturation slider sits on
    the right.  Any change emits the chosen BGR tuple via ``on_color``.
    """

    def __init__(self, master, on_color=None, height: int = 72) -> None:
        super().__init__(master)
        self.on_color = on_color
        self._height = height
        self._hue = 0.0      # 0..360
        self._value = 100.0  # 0..100
        self._sat = 100.0    # 0..100
        self._canvas = tk.Canvas(self, height=height, cursor="crosshair")
        self._canvas.pack(side="left", fill="x", expand=True)
        self._canvas.bind("<Button-1>", self._on_pick)
        self._canvas.bind("<B1-Motion>", self._on_pick)
        ttk.Scale(self, from_=0, to=100, orient="vertical", length=height,
                  command=self._on_sat).pack(side="right")
        self.after_idle(self._redraw)

    def _redraw(self) -> None:
        w = self._canvas.winfo_width() or 180
        h = self._height
        for x in range(0, max(1, w), 2):
            hue = (x / max(1, w)) * 360.0
            for y in range(0, h, 2):
                value = 100.0 - (y / h) * 100.0
                b, g, r = ip.hsv_to_bgr(hue, self._sat, value)
                hexc = "#%02x%02x%02x" % (r, g, b)
                self._canvas.create_rectangle(x, y, x + 2, y + 2, fill=hexc, outline="")

    def _emit(self) -> None:
        if self.on_color:
            self.on_color(ip.hsv_to_bgr(self._hue, self._sat, self._value))

    def _on_pick(self, event) -> None:
        w = self._canvas.winfo_width() or 1
        h = self._height
        self._hue = (float(event.x) / w) * 360.0
        self._value = 100.0 - (float(event.y) / h) * 100.0
        self._emit()

    def _on_sat(self, value: str) -> None:
        self._sat = float(value)
        self._redraw()
        self._emit()
