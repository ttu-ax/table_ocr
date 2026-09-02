"""Pure image-processing logic for the preprocessing tab.

This module is UI-free: four-point document perspective correction and an
RGB overlay painter used by ``preprocess_frame`` (and reused by the
recognition pipeline).  Everything operates on BGR ``numpy`` arrays so it can be
unit-tested without a window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Four-point detection + perspective correction
# --------------------------------------------------------------------------- #
@dataclass
class Quad:
    """Four ordered corners (top-left, top-right, bottom-right, bottom-left)."""
    points: np.ndarray  # float32 array shape (4, 2), in original pixel coords

    @property
    def ordered(self) -> np.ndarray:
        return self.points

    def to_list(self) -> list[list[float]]:
        return self.points.tolist()


def order_quad(points: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL regardless of input order."""
    points = points.astype(np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]          # TL: min x+y
    ordered[2] = points[np.argmax(sums)]          # BR: max x+y
    ordered[1] = points[np.argmin(differences)]   # TR: min y-x
    ordered[3] = points[np.argmax(differences)]   # BL: max y-x
    return ordered


def detect_document_quad(image: np.ndarray) -> Quad | None:
    """Detect the outer table/document quad on a photographed page.

    Returns normalized original-coordinate corners, or ``None`` if no reliable
    quadrilateral is found.
    """
    original_height, original_width = image.shape[:2]
    detection_scale = min(1.0, 2200.0 / max(original_height, original_width))
    small = cv2.resize(
        image,
        None,
        fx=detection_scale,
        fy=detection_scale,
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15
    )
    height, width = binary.shape
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(35, width // 24), 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(35, height // 24))),
    )
    grid = cv2.bitwise_or(horizontal, vertical)
    grid = cv2.morphologyEx(
        grid,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=2,
    )
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = width * height * 0.20
    quad: np.ndarray | None = None
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in sorted_contours:
        if cv2.contourArea(contour) < minimum_area:
            break
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        approximation = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            quad = approximation.reshape(4, 2).astype(np.float32) / detection_scale
            break
    if quad is None and sorted_contours and cv2.contourArea(sorted_contours[0]) >= minimum_area:
        box = cv2.boxPoints(cv2.minAreaRect(sorted_contours[0]))
        quad = box.astype(np.float32) / detection_scale

    if quad is None:
        return None

    # Expand a touch so edge content isn't clipped, then bound to the image.
    center = quad.mean(axis=0)
    quad = center + (quad - center) * 1.05
    quad[:, 0] = np.clip(quad[:, 0], 0, original_width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, original_height - 1)
    return Quad(order_quad(quad))


def warp_document(image: np.ndarray, quad: Quad, add_white_border: bool = False) -> np.ndarray:
    """Perspective-warp the selected quad to an upright rectangle.

    By default the output is exactly the selected content (no border), so there
    are no black/blank edges.  ``add_white_border`` optionally adds a small white
    margin around the result.
    """
    source = quad.ordered
    target_width = int(max(
        np.linalg.norm(source[1] - source[0]),
        np.linalg.norm(source[2] - source[3]),
    ))
    target_height = int(max(
        np.linalg.norm(source[3] - source[0]),
        np.linalg.norm(source[2] - source[1]),
    ))
    if target_width < 10 or target_height < 10:
        return image

    destination = np.array(
        [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        image,
        transform,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    if add_white_border:
        warped = cv2.copyMakeBorder(
            warped, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
    return warped


# --------------------------------------------------------------------------- #
# Overlay painter
# --------------------------------------------------------------------------- #
@dataclass
class Stroke:
    """One freehand paint stroke in original-image coordinates."""
    color: tuple[int, int, int]  # BGR
    radius: int
    points: list[tuple[float, float]] = field(default_factory=list)


class Painter:
    """Applies a list of strokes onto a BGR image as a pure overlay.

    Strokes are recorded in *original* image coordinates (the pre-correction
    image), then composited before any perspective warp, so the painted content
    is carried through the correction.
    """

    def __init__(self) -> None:
        self.strokes: list[Stroke] = []

    def clear(self) -> None:
        self.strokes.clear()

    def add_stroke(self, stroke: Stroke) -> None:
        self.strokes.append(stroke)

    def pop(self) -> Stroke | None:
        return self.strokes.pop() if self.strokes else None

    @property
    def empty(self) -> bool:
        return not self.strokes

    def composite(self, image: np.ndarray) -> np.ndarray:
        """Return a copy of ``image`` with all strokes painted on (BGR)."""
        out = image.copy()
        for stroke in self.strokes:
            if not stroke.points:
                continue
            color = tuple(int(ch) for ch in stroke.color)
            pts = np.array([[int(round(x)), int(round(y))] for x, y in stroke.points], np.int32)
            cv2.polylines(
                out, [pts], isClosed=False, color=color,
                thickness=int(stroke.radius * 2), lineType=cv2.LINE_AA,
            )
            for center in stroke.points:
                cv2.circle(
                    out, (int(round(center[0])), int(round(center[1]))),
                    int(stroke.radius), color, -1, lineType=cv2.LINE_AA,
                )
        return out


# --------------------------------------------------------------------------- #
# Color helpers
# --------------------------------------------------------------------------- #
def hsv_to_bgr(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Convert HSV (h 0..360, s/v 0..100) to a BGR tuple for OpenCV drawing.

    Hue/sat/value are clamped to their nominal ranges so boundary values (e.g.
    hue=360 or sat=100) never overflow an 8-bit channel.
    """
    h = float(np.clip(h, 0.0, 359.0))
    s = float(np.clip(s, 0.0, 100.0))
    v = float(np.clip(v, 0.0, 100.0))
    hsv = np.uint8([[[int(round(h / 2.0)), int(round(s * 255.0 / 100.0)), int(round(v * 255.0 / 100.0))]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def rgb_str(bgr: tuple[int, int, int]) -> str:
    """Format a BGR tuple as a human-readable RGB string."""
    b, g, r = bgr
    return f"RGB({r}, {g}, {b})"
