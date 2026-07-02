from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps, ImageTk
except ImportError as exc:  # pragma: no cover - startup guard
    raise SystemExit(
        "Pillow is required. Run: python -m pip install -r requirements.txt"
    ) from exc


NUMBER_RE = re.compile(r"(?<!\d)(\d{1,4})(?!\d)")
CONFIDENCE_RE = re.compile(r"\b(0\.\d{1,4}|1\.0{1,4})\b")
ARTICLE_LINE_RE = re.compile(
    r"^\s*(?:№|N|No\.?|Поз\.?|Позиция)?\s*(\d{1,4})\s*[\).\]:;\-–—]?\s+(.+?)\s*$",
    re.IGNORECASE,
)
WORK_IMAGE_SIZE = 800
HQ_MIN_SQUARE_SIZE = WORK_IMAGE_SIZE
HQ_MAX_SQUARE_SIZE = 12000
HQ_RENDER_TARGET_LONG_SIDE = 7200
HQ_RENDER_MAX_SCALE = 12.0
WORK_TILE_SIZE = 96
FINAL_TILE_SIZE = 64
FINAL_SUPERSAMPLE = 3
APP_BUILD = "2026-06-19 09:18 v4.17 fragments"
GRID_OCR_PASSES = (
    (256, 128),
    (320, 160),
    (400, 200),
)
OPENCV_MULTISCALE_SIZES = (1200, 1600, 2400)


def application_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


@dataclass
class Hotspot:
    number: str
    x: float
    y: float
    width: float
    height: float
    article: str = ""
    source: str = "ocr"

    def center_percent(self, image_width: int, image_height: int) -> tuple[float, float]:
        return self.x / image_width * 100, self.y / image_height * 100

    def area_percent(self, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        click_width = max(self.width * 1.7, 28)
        click_height = max(self.height * 1.7, 28)
        left = max(0, self.x - click_width / 2)
        top = max(0, self.y - click_height / 2)
        right = min(image_width, left + click_width)
        bottom = min(image_height, top + click_height)
        return (
            left / image_width * 100,
            top / image_height * 100,
            (right - left) / image_width * 100,
            (bottom - top) / image_height * 100,
        )


@dataclass
class CutoutRegion:
    box: tuple[int, int, int, int]
    mask: Image.Image | None = None


@dataclass
class ImageFragment:
    name: str
    image: Image.Image
    image_path: Path
    source_image_size: tuple[int, int]
    source_to_work_scale: float
    source_to_work_offset_x: float
    source_to_work_offset_y: float
    pdf_source_to_work_scale: float
    pdf_source_to_work_offset_x: float
    pdf_source_to_work_offset_y: float
    current_image_is_pdf_page: bool
    raster_source_image: Image.Image | None
    spots: list[Hotspot]
    brush_strokes: list[tuple[float, float, float, float, float]]
    cutout_regions: list[CutoutRegion]


@dataclass
class LabelLineBox:
    crop: tuple[int, int, int, int]
    line_x: int
    line_y: int
    line_width: int
    line_height: int


@dataclass
class LeaderLineSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    source: str = "hough"


@dataclass
class LabelMontagePlacement:
    montage_x: int
    montage_y: int
    scale: float
    source_box: tuple[int, int, int, int]
    width: int
    height: int

    def contains(self, x: float, y: float) -> bool:
        return self.montage_x <= x <= self.montage_x + self.width and self.montage_y <= y <= self.montage_y + self.height

    def to_image_point(self, x: float, y: float) -> tuple[float, float]:
        source_x, source_y, _right, _bottom = self.source_box
        return (
            source_x + (x - self.montage_x) / self.scale,
            source_y + (y - self.montage_y) / self.scale,
        )


class OcrError(RuntimeError):
    pass


def normalize_number(value: str) -> str:
    match = NUMBER_RE.search(value or "")
    if not match:
        return (value or "").strip()
    digits = match.group(1)
    return str(int(digits)) if digits.isdigit() else digits


def get_article_for_number(articles: dict[str, str], number: str) -> str:
    return articles.get(number, articles.get(normalize_number(number), ""))


def resolve_ocr_number(value: str, known_numbers: set[str]) -> str | None:
    normalized = normalize_number(value)
    if not known_numbers:
        if normalized == "0":
            return None
        if normalized.isdigit() and int(normalized) > 500:
            return None
        return normalized
    if normalized in known_numbers:
        return normalized

    # Sometimes OCR glues a leader line or nearby stroke to the number:
    # "32" can become "321". Prefer a known multi-digit prefix/suffix.
    prefix_candidates = [
        known
        for known in known_numbers
        if len(known) >= 2 and normalized.startswith(known)
    ]
    if prefix_candidates:
        return max(prefix_candidates, key=len)

    suffix_candidates = [
        known
        for known in known_numbers
        if len(known) >= 2 and normalized.endswith(known)
    ]
    if not suffix_candidates:
        return None
    return max(suffix_candidates, key=len)


def source_confidence(source: str) -> float | None:
    values = [float(match.group(1)) for match in CONFIDENCE_RE.finditer(source or "")]
    if not values:
        return None
    return max(value for value in values if 0.0 <= value <= 1.0)


def render_square_from_source_tiles(
    image: Image.Image,
    size: int = WORK_IMAGE_SIZE,
    tile_size: int = WORK_TILE_SIZE,
    tile_overlap: int = 8,
    sharpen: bool = True,
    sharpen_radius: float = 0.85,
    sharpen_percent: int = 180,
    sharpen_threshold: int = 2,
) -> tuple[Image.Image, float, float, float]:
    source = image.convert("RGB")
    width, height = source.size
    if width <= 0 or height <= 0:
        return Image.new("RGB", (size, size), "white"), 1.0, 0.0, 0.0

    scale = min(size / width, size / height)
    content_width = max(1, int(round(width * scale)))
    content_height = max(1, int(round(height * scale)))
    offset_x = float(int(round((size - content_width) / 2)))
    offset_y = float(int(round((size - content_height) / 2)))
    result = Image.new("RGB", (size, size), "white")
    tile_size = max(32, int(tile_size))
    tile_overlap = max(0, int(tile_overlap))
    scale_x = content_width / max(width, 1)
    scale_y = content_height / max(height, 1)
    left = int(offset_x)
    top = int(offset_y)
    right = left + content_width
    bottom = top + content_height

    for tile_top in range(top, bottom, tile_size):
        tile_bottom = min(tile_top + tile_size, bottom)
        for tile_left in range(left, right, tile_size):
            tile_right = min(tile_left + tile_size, right)
            tile_width = tile_right - tile_left
            tile_height = tile_bottom - tile_top
            if tile_width <= 0 or tile_height <= 0:
                continue
            expanded_left = max(left, tile_left - tile_overlap)
            expanded_top = max(top, tile_top - tile_overlap)
            expanded_right = min(right, tile_right + tile_overlap)
            expanded_bottom = min(bottom, tile_bottom + tile_overlap)
            expanded_width = expanded_right - expanded_left
            expanded_height = expanded_bottom - expanded_top
            if expanded_width <= 0 or expanded_height <= 0:
                continue

            source_left = (expanded_left - left) / max(scale_x, 0.0001)
            source_top = (expanded_top - top) / max(scale_y, 0.0001)
            source_right = (expanded_right - left) / max(scale_x, 0.0001)
            source_bottom = (expanded_bottom - top) / max(scale_y, 0.0001)
            source_box = (
                max(0, int(math.floor(source_left))),
                max(0, int(math.floor(source_top))),
                min(width, int(math.ceil(source_right))),
                min(height, int(math.ceil(source_bottom))),
            )
            if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
                continue
            expanded_tile = source.crop(source_box).resize(
                (expanded_width, expanded_height),
                Image.Resampling.LANCZOS,
            )
            tile = expanded_tile.crop(
                (
                    tile_left - expanded_left,
                    tile_top - expanded_top,
                    tile_right - expanded_left,
                    tile_bottom - expanded_top,
                )
            )
            result.paste(tile, (tile_left, tile_top))

    if sharpen:
        result = result.filter(
            ImageFilter.UnsharpMask(
                radius=sharpen_radius,
                percent=sharpen_percent,
                threshold=sharpen_threshold,
            )
        )
    return result, scale, offset_x, offset_y


def normalize_to_work_square(
    image: Image.Image,
    size: int = WORK_IMAGE_SIZE,
    sharpen: bool = True,
) -> tuple[Image.Image, float, float, float]:
    return render_square_from_source_tiles(
        image,
        size=size,
        sharpen=sharpen,
        tile_size=WORK_TILE_SIZE,
        tile_overlap=8,
        sharpen_radius=1.0,
        sharpen_percent=155,
        sharpen_threshold=2,
    )


def tile_positions(length: int, tile_size: int, step: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


class WindowsOcrBackend:
    name = "Windows OCR"

    def available(self) -> bool:
        try:
            import winsdk  # noqa: F401
        except Exception:
            return False
        return True

    def recognize_digits(self, image_path: Path, known_numbers: set[str] | None = None) -> list[Hotspot]:
        if not self.available():
            raise OcrError("winsdk не установлен")
        return asyncio.run(self._recognize_digits_with_tiles_async(image_path))

    async def _recognize_digits_with_tiles_async(self, image_path: Path) -> list[Hotspot]:
        spots = await self._recognize_digits_async(image_path)

        try:
            with Image.open(image_path) as image:
                width, height = image.size
                if width < 350 or height < 250:
                    return _dedupe_spots(spots)
                boxes = self._ocr_tile_boxes(width, height)
                for index, box in enumerate(boxes):
                    left, top, right, bottom = box
                    crop = image.crop(box).convert("RGB")
                    tile_path = Path(tempfile.gettempdir()) / f"parts_ocr_tile_{os.getpid()}_{index}.png"
                    crop.save(tile_path)
                    try:
                        tile_spots = await self._recognize_digits_async(tile_path)
                    finally:
                        tile_path.unlink(missing_ok=True)
                    for spot in tile_spots:
                        spot.x += left
                        spot.y += top
                    spots.extend(tile_spots)
        except Exception:
            return _dedupe_spots(spots)

        return _dedupe_spots(spots)

    def _ocr_tile_boxes(self, width: int, height: int) -> list[tuple[int, int, int, int]]:
        boxes = [
            (0, int(height * 0.16), width, int(height * 0.42)),
            (int(width * 0.35), int(height * 0.84), int(width * 0.90), min(height, int(height * 0.99))),
            (int(width * 0.38), int(height * 0.86), int(width * 0.86), min(height, int(height * 0.985))),
            (int(width * 0.68), int(height * 0.06), min(width, int(width * 0.98)), int(height * 0.34)),
        ]
        if width >= 900 and height >= 650:
            for top_ratio, bottom_ratio in ((0.0, 0.55), (0.45, 1.0)):
                for left_ratio, right_ratio in ((0.0, 0.42), (0.29, 0.71), (0.58, 1.0)):
                    boxes.append(
                        (
                            int(width * left_ratio),
                            int(height * top_ratio),
                            int(width * right_ratio),
                            int(height * bottom_ratio),
                        )
                    )

        result: list[tuple[int, int, int, int]] = []
        for left, top, right, bottom in boxes:
            left = max(0, min(left, width - 1))
            top = max(0, min(top, height - 1))
            right = max(left + 40, min(right, width))
            bottom = max(top + 40, min(bottom, height))
            box = (left, top, right, bottom)
            if right <= width and bottom <= height and box not in result:
                result.append(box)
        return result

    async def _recognize_digits_async(self, image_path: Path) -> list[Hotspot]:
        from winsdk.windows.graphics.imaging import BitmapDecoder
        from winsdk.windows.media.ocr import OcrEngine
        from winsdk.windows.storage import FileAccessMode, StorageFile

        ocr_path, scale = self._prepare_image_for_windows_ocr(image_path)
        try:
            file = await StorageFile.get_file_from_path_async(str(ocr_path))
            stream = await file.open_async(FileAccessMode.READ)
            decoder = await BitmapDecoder.create_async(stream)
            bitmap = await decoder.get_software_bitmap_async()
            engine = OcrEngine.try_create_from_user_profile_languages()
            if engine is None:
                raise OcrError("Windows OCR не смог создать движок распознавания")
            result = await engine.recognize_async(bitmap)
        finally:
            if ocr_path != image_path:
                try:
                    ocr_path.unlink(missing_ok=True)
                except Exception:
                    pass

        spots: list[Hotspot] = []
        for line in result.lines:
            for word in line.words:
                text = getattr(word, "text", "") or ""
                rect = word.bounding_rect
                for number in NUMBER_RE.findall(text):
                    x = (rect.x + rect.width / 2) / scale
                    y = (rect.y + rect.height / 2) / scale
                    spots.append(
                        Hotspot(
                            number=number,
                            x=x,
                            y=y,
                            width=max(rect.width / scale, 1),
                            height=max(rect.height / scale, 1),
                            source=self.name,
                        )
                    )
        return _dedupe_spots(spots)

    def _prepare_image_for_windows_ocr(self, image_path: Path) -> tuple[Path, float]:
        # Windows OCR can reject very large bitmaps and often misses tiny labels.
        # Resize only the temporary OCR copy, then scale coordinates back.
        max_side = 2400
        target_side = 1600
        with Image.open(image_path) as image:
            width, height = image.size
            largest = max(width, height)
            if target_side <= largest <= max_side:
                return image_path, 1.0

            if largest < target_side:
                scale = min(max_side / largest, target_side / largest)
            else:
                scale = max_side / largest
            new_size = (int(width * scale), int(height * scale))
            resized = image.convert("RGB").resize(new_size, Image.Resampling.LANCZOS)
            tmp = Path(tempfile.gettempdir()) / f"parts_ocr_{os.getpid()}_{image_path.stem}.png"
            resized.save(tmp)
            return tmp, scale


class TesseractBackend:
    name = "Tesseract"

    def available(self) -> bool:
        if shutil.which("tesseract") is None:
            return False
        try:
            import pytesseract  # noqa: F401
        except Exception:
            return False
        return True

    def recognize_digits(self, image_path: Path, known_numbers: set[str] | None = None) -> list[Hotspot]:
        if not self.available():
            raise OcrError("Tesseract не установлен или не найден в PATH")

        import pytesseract
        from pytesseract import Output

        with Image.open(image_path) as image:
            data = pytesseract.image_to_data(
                image,
                config="--psm 6 -c tessedit_char_whitelist=0123456789",
                output_type=Output.DICT,
            )

        spots: list[Hotspot] = []
        for i, text in enumerate(data.get("text", [])):
            for number in NUMBER_RE.findall(text or ""):
                left = float(data["left"][i])
                top = float(data["top"][i])
                width = max(float(data["width"][i]), 1)
                height = max(float(data["height"][i]), 1)
                spots.append(
                    Hotspot(
                        number=number,
                        x=left + width / 2,
                        y=top + height / 2,
                        width=width,
                        height=height,
                        source=self.name,
                    )
                )
        return _dedupe_spots(spots)


class GridTileOcrBackend:
    name = "800 grid OCR"

    def available(self) -> bool:
        return WindowsOcrBackend().available()

    def recognize_digits(self, image_path: Path, known_numbers: set[str] | None = None) -> list[Hotspot]:
        if not self.available():
            raise OcrError("Windows OCR недоступен")

        with Image.open(image_path) as image:
            work_image = image.convert("RGB").resize((WORK_IMAGE_SIZE, WORK_IMAGE_SIZE), Image.Resampling.LANCZOS)

        spots: list[Hotspot] = []
        backend = WindowsOcrBackend()
        full_path = Path(tempfile.gettempdir()) / f"parts_grid_full_{os.getpid()}_{image_path.stem}.png"
        work_image.save(full_path)
        try:
            spots.extend(backend.recognize_digits(full_path, known_numbers))
        finally:
            full_path.unlink(missing_ok=True)

        tile_index = 0
        for tile_size, step in GRID_OCR_PASSES:
            for top in tile_positions(WORK_IMAGE_SIZE, tile_size, step):
                for left in tile_positions(WORK_IMAGE_SIZE, tile_size, step):
                    box = (left, top, left + tile_size, top + tile_size)
                    tile = work_image.crop(box)
                    tile_path = (
                        Path(tempfile.gettempdir())
                        / f"parts_grid_tile_{os.getpid()}_{image_path.stem}_{tile_size}_{tile_index}.png"
                    )
                    tile_index += 1
                    tile.save(tile_path)
                    try:
                        tile_spots = backend.recognize_digits(tile_path, known_numbers)
                    finally:
                        tile_path.unlink(missing_ok=True)
                    for spot in tile_spots:
                        spot.x += left
                        spot.y += top
                        spot.source = f"{self.name} {tile_size}"
                    spots.extend(tile_spots)

        return _dedupe_spots(spots)


class LineLabelWindowsBackend:
    name = "Line label OCR"

    def available(self) -> bool:
        return WindowsOcrBackend().available() and _opencv_available()

    def recognize_digits(self, image_path: Path, known_numbers: set[str] | None = None) -> list[Hotspot]:
        if not self.available():
            raise OcrError("Windows OCR или OpenCV недоступен")

        boxes = _find_horizontal_label_boxes(image_path)
        if not boxes:
            return []

        montage_path, placements = _build_label_montage(image_path, boxes)
        try:
            spots = WindowsOcrBackend().recognize_digits(montage_path, known_numbers)
        finally:
            montage_path.unlink(missing_ok=True)

        mapped: list[Hotspot] = []
        for spot in spots:
            for placement in placements:
                if not placement.contains(spot.x, spot.y):
                    continue
                x, y = placement.to_image_point(spot.x, spot.y)
                mapped.append(
                    Hotspot(
                        number=spot.number,
                        x=x,
                        y=y,
                        width=max(spot.width / placement.scale, 1),
                        height=max(spot.height / placement.scale, 1),
                        source=self.name,
                    )
                )
                break
        return _dedupe_spots(mapped)


class LineLabelDigitBackend:
    name = "Line label detector"

    def available(self) -> bool:
        return _opencv_available()

    def recognize_digits(self, image_path: Path, known_numbers: set[str] | None = None) -> list[Hotspot]:
        if not self.available():
            raise OcrError("OpenCV не установлен")

        import cv2

        known_numbers = known_numbers or {str(index) for index in range(1, 500)}
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise OcrError("Не удалось прочитать изображение")

        boxes = _find_horizontal_label_boxes(image_path, gray=gray)
        if not boxes:
            return []

        max_digits = max((len(number) for number in known_numbers if number.isdigit()), default=3)
        spots: list[Hotspot] = []
        digit_backend = OpenCVDigitBackend()
        for box in boxes:
            spots.extend(
                self._recognize_near_label_line(
                    gray,
                    box,
                    known_numbers,
                    max_digits,
                    digit_backend,
                )
            )

        return _dedupe_spots(spots)

    def _recognize_near_label_line(
        self,
        gray,
        box: LabelLineBox,
        known_numbers: set[str],
        max_digits: int,
        digit_backend: "OpenCVDigitBackend",
    ) -> list[Hotspot]:
        import cv2

        spots: list[Hotspot] = []
        image_height, image_width = gray.shape[:2]
        for mode in ("above", "below"):
            if mode == "above":
                left = max(0, box.line_x - 8)
                right = min(image_width, box.line_x + box.line_width + 8)
                top = max(0, box.line_y - 58)
                bottom = max(0, box.line_y - 1)
            else:
                left = max(0, box.line_x - 8)
                right = min(image_width, box.line_x + box.line_width + 8)
                top = min(image_height, box.line_y + box.line_height + 1)
                bottom = min(image_height, box.line_y + box.line_height + 58)

            if bottom - top < 10 or right - left < 10:
                continue

            roi = gray[top:bottom, left:right]
            _, binary = cv2.threshold(roi, 170, 255, cv2.THRESH_BINARY_INV)
            line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            line_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, line_kernel)
            binary = cv2.subtract(binary, line_mask)

            count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
            candidates: list[dict[str, float | str]] = []
            for label in range(1, count):
                x = int(stats[label, cv2.CC_STAT_LEFT])
                y = int(stats[label, cv2.CC_STAT_TOP])
                component_width = int(stats[label, cv2.CC_STAT_WIDTH])
                component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
                area = int(stats[label, cv2.CC_STAT_AREA])
                if not _line_label_component_size_ok(component_width, component_height, area):
                    continue

                digit, score = _recognize_line_label_digit(
                    binary[y : y + component_height, x : x + component_width]
                )
                if digit is None:
                    continue

                candidates.append(
                    {
                        "digit": digit,
                        "score": score,
                        "x": float(left + x),
                        "y": float(top + y),
                        "w": float(component_width),
                        "h": float(component_height),
                        "cx": float(left + x + component_width / 2),
                        "cy": float(top + y + component_height / 2),
                    }
                )

            for group in digit_backend._group_digits(candidates):
                if not (1 <= len(group) <= max_digits):
                    continue
                number = normalize_number("".join(str(item["digit"]) for item in group))
                if known_numbers and number not in known_numbers:
                    continue

                group_left = min(float(item["x"]) for item in group)
                group_top = min(float(item["y"]) for item in group)
                group_right = max(float(item["x"]) + float(item["w"]) for item in group)
                group_bottom = max(float(item["y"]) + float(item["h"]) for item in group)
                center_x = (group_left + group_right) / 2
                if center_x < box.line_x - 15 or center_x > box.line_x + box.line_width + 15:
                    continue

                spots.append(
                    Hotspot(
                        number=number,
                        x=center_x,
                        y=(group_top + group_bottom) / 2,
                        width=max(group_right - group_left, 1),
                        height=max(group_bottom - group_top, 1),
                        source=self.name,
                    )
                )

        return spots


class LeaderEndpointOcrBackend:
    name = "Leader endpoint OCR"

    def available(self) -> bool:
        return _opencv_available() and WindowsOcrBackend().available()

    def recognize_digits(self, image_path: Path, known_numbers: set[str] | None = None) -> list[Hotspot]:
        if not self.available():
            raise OcrError("Windows OCR или OpenCV недоступны")

        endpoints = _find_leader_line_endpoints(image_path, max_points=100)
        if not endpoints:
            return []

        montage_path, placements = _build_endpoint_montage(image_path, endpoints)
        try:
            raw_spots: list[Hotspot] = []
            try:
                raw_spots.extend(WindowsOcrBackend().recognize_digits(montage_path, known_numbers))
            except Exception:
                pass
            try:
                raw_spots.extend(OpenCVDigitBackend().recognize_digits(montage_path, known_numbers))
            except Exception:
                pass
        finally:
            montage_path.unlink(missing_ok=True)

        mapped: list[Hotspot] = []
        for spot in raw_spots:
            for placement in placements:
                if not placement.contains(spot.x, spot.y):
                    continue
                x, y = placement.to_image_point(spot.x, spot.y)
                mapped.append(
                    Hotspot(
                        number=spot.number,
                        x=x,
                        y=y,
                        width=max(spot.width / placement.scale, 1),
                        height=max(spot.height / placement.scale, 1),
                        source=f"{self.name}; {spot.source}",
                    )
                )
                break
        return _dedupe_spots(mapped)


class DenseCropWindowsOcrBackend:
    name = "Dense crop Windows OCR"

    def available(self) -> bool:
        return WindowsOcrBackend().available()

    def recognize_digits(self, image_path: Path, known_numbers: set[str] | None = None) -> list[Hotspot]:
        if not self.available():
            raise OcrError("Windows OCR недоступен")

        with Image.open(image_path) as image:
            source = image.convert("RGB")
        width, height = source.size
        if width < 320 or height < 320:
            return []

        backend = WindowsOcrBackend()
        spots: list[Hotspot] = []
        for index, box in enumerate(self._dense_crop_boxes(width, height)):
            left, top, right, bottom = box
            if right - left < 120 or bottom - top < 120:
                continue

            crop_path = Path(tempfile.gettempdir()) / f"parts_dense_crop_{os.getpid()}_{index}.png"
            source.crop(box).save(crop_path)
            try:
                crop_spots = backend.recognize_digits(crop_path, known_numbers)
            except Exception:
                crop_spots = []
            finally:
                crop_path.unlink(missing_ok=True)

            for spot in crop_spots:
                spot.x += left
                spot.y += top
                spot.source = f"{self.name}; {spot.source}"
                spots.append(spot)

        return _dedupe_spots(spots)

    def _dense_crop_boxes(self, width: int, height: int) -> list[tuple[int, int, int, int]]:
        ratios = (
            (0.64, 0.25, 0.94, 0.45),
            (0.64, 0.22, 0.98, 0.46),
            (0.50, 0.20, 0.85, 0.45),
            (0.35, 0.20, 0.70, 0.45),
            (0.15, 0.28, 0.55, 0.55),
            (0.30, 0.35, 0.70, 0.60),
        )
        boxes: list[tuple[int, int, int, int]] = []
        for left_ratio, top_ratio, right_ratio, bottom_ratio in ratios:
            box = (
                max(0, min(width - 1, int(width * left_ratio))),
                max(0, min(height - 1, int(height * top_ratio))),
                max(1, min(width, int(width * right_ratio))),
                max(1, min(height, int(height * bottom_ratio))),
            )
            if box[2] > box[0] and box[3] > box[1] and box not in boxes:
                boxes.append(box)
        return boxes


class CircledNumberBackend:
    name = "Circled number detector"

    def available(self) -> bool:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except Exception:
            return False
        return True

    def recognize_digits(self, image_path: Path, known_numbers: set[str] | None = None) -> list[Hotspot]:
        if not self.available():
            raise OcrError("OpenCV не установлен")

        import cv2
        import numpy as np

        known_numbers = known_numbers or {str(index) for index in range(1, 100)}
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise OcrError("Не удалось прочитать изображение")

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return []

        spots: list[Hotspot] = []
        for index, contour in enumerate(contours):
            x, y, width, height = cv2.boundingRect(contour)
            if not self._circle_candidate_ok(contour, width, height, hierarchy[0][index]):
                continue

            result = self._classify_circle(binary[y : y + height, x : x + width], known_numbers)
            if result is None:
                continue

            number, score = result
            spots.append(
                Hotspot(
                    number=number,
                    x=x + width / 2,
                    y=y + height / 2,
                    width=width,
                    height=height,
                    source=f"{self.name} {score:.2f}",
                )
            )

        return _dedupe_spots(spots)

    def _circle_candidate_ok(self, contour, width: int, height: int, hierarchy_row) -> bool:
        import cv2
        import numpy as np

        if width < 9 or height < 9 or width > 24 or height > 24:
            return False
        aspect = width / max(height, 1)
        if aspect < 0.65 or aspect > 1.55:
            return False

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0
        child = int(hierarchy_row[2])
        if area < 45 or circularity < 0.60:
            return False

        return child != -1 or (area >= 80 and circularity >= 0.78)

    def _classify_circle(self, binary_crop, known_numbers: set[str]) -> tuple[str, float] | None:
        import numpy as np

        digit_crop = _mask_circle_inner(binary_crop)
        if int(np.sum(digit_crop > 0)) < 2:
            return None

        vector = (digit_crop > 0).astype("float32").ravel()
        vector -= vector.mean()
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-6:
            return None
        vector /= norm

        matrix, labels = _circled_number_template_matrix(
            binary_crop.shape[1],
            binary_crop.shape[0],
            tuple(sorted(known_numbers, key=_number_sort_key)),
        )
        if len(labels) == 0:
            return None

        scores = matrix @ vector
        best_by_label: dict[str, float] = {}
        for score, label in zip(scores, labels):
            best_by_label[label] = max(best_by_label.get(label, -999.0), float(score))

        ranked = sorted(best_by_label.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            return None

        label, score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -999.0
        margin = score - second_score
        if label == "7" and "2" in known_numbers and _circled_digit_looks_like_two(digit_crop):
            label = "2"
        if score >= 0.76 or (score >= 0.66 and margin >= 0.045):
            return label, score
        if label == "7" and score >= 0.72 and not _circled_digit_looks_like_two(digit_crop):
            return label, score
        return None


def _number_sort_key(value: str) -> int:
    return int(value) if value.isdigit() else 999999


def _mask_circle_inner(binary_crop):
    import numpy as np

    height, width = binary_crop.shape[:2]
    yy, xx = np.indices(binary_crop.shape)
    cx = (width - 1) / 2
    cy = (height - 1) / 2
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    result = binary_crop.copy()
    result[distance > min(width, height) * 0.37] = 0
    return result


def _circled_digit_looks_like_two(digit_crop) -> bool:
    import numpy as np

    height, width = digit_crop.shape[:2]
    ys, xs = np.where(digit_crop > 0)
    if len(xs) == 0:
        return False

    center = (width - 1) / 2
    lower_half = ys >= height * 0.58
    lower_left_pixels = xs[lower_half] <= center - 1.5
    bottom_pixels = ys >= height * 0.68
    return int(np.sum(lower_left_pixels)) >= 2 and int(np.sum(bottom_pixels)) >= 4


@lru_cache(maxsize=128)
def _circled_number_template_matrix(width: int, height: int, known_numbers: tuple[str, ...]):
    import cv2
    import numpy as np

    font_names = ("arial.ttf", "arialbd.ttf", "calibri.ttf", "calibrib.ttf", "tahoma.ttf")
    fonts = []
    for font_name in font_names:
        for size in range(7, 14):
            try:
                fonts.append(ImageFont.truetype(font_name, size))
            except Exception:
                pass
    if not fonts:
        fonts.append(ImageFont.load_default())

    rows = []
    labels = []
    for number in known_numbers:
        for font in fonts:
            for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                image = Image.new("L", (width, height), 0)
                draw = ImageDraw.Draw(image)
                bbox = draw.textbbox((0, 0), number, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                draw.text(
                    (
                        (width - text_width) / 2 - bbox[0] + dx,
                        (height - text_height) / 2 - bbox[1] + dy,
                    ),
                    number,
                    fill=255,
                    font=font,
                )
                array = np.array(image)
                _, template = cv2.threshold(array, 32, 255, cv2.THRESH_BINARY)
                template = _mask_circle_inner(template)
                if int(np.sum(template > 0)) == 0:
                    continue

                vector = (template > 0).astype("float32").ravel()
                vector -= vector.mean()
                norm = float(np.linalg.norm(vector))
                if norm <= 1e-6:
                    continue
                rows.append(vector / norm)
                labels.append(number)

    matrix = np.vstack(rows) if rows else np.zeros((0, width * height), dtype="float32")
    return matrix, tuple(labels)


def _opencv_available() -> bool:
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except Exception:
        return False
    return True


def _find_horizontal_label_boxes(image_path: Path, gray=None) -> list[LabelLineBox]:
    import cv2

    if gray is None:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return []

    _, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1)),
    )
    horizontal = cv2.dilate(
        horizontal,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)),
        iterations=1,
    )
    contours, _hierarchy = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_height, image_width = gray.shape[:2]
    boxes: list[LabelLineBox] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < 35 or width > 360 or height > 12:
            continue
        crop = (
            max(0, x - 8),
            max(0, y - 60),
            min(image_width, x + width + 8),
            min(image_height, y + height + 18),
        )
        boxes.append(
            LabelLineBox(
                crop=crop,
                line_x=x,
                line_y=y,
                line_width=width,
                line_height=height,
            )
        )

    boxes.sort(key=lambda item: (item.line_y, item.line_x))
    result: list[LabelLineBox] = []
    for box in boxes:
        if any(abs(box.line_x - existing.line_x) < 8 and abs(box.line_y - existing.line_y) < 8 for existing in result):
            continue
        result.append(box)
    return result


def _find_leader_line_endpoints(image_path: Path, max_points: int = 120) -> list[tuple[float, float]]:
    import cv2
    import numpy as np

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return []

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 60, 180)
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=max(14, min(gray.shape[:2]) // 60),
        maxLineGap=6,
    )

    candidates: list[tuple[float, float, float]] = []
    if raw_lines is not None:
        for raw_line in raw_lines[:, 0]:
            x1, y1, x2, y2 = [float(value) for value in raw_line]
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if length < 18 or length > max(gray.shape[:2]) * 0.55:
                continue
            candidates.append((x1, y1, length))
            candidates.append((x2, y2, length))

    for box in _find_horizontal_label_boxes(image_path, gray=gray):
        y = float(box.line_y + box.line_height / 2)
        length = float(box.line_width)
        candidates.append((float(box.line_x), y, length + 20))
        candidates.append((float(box.line_x + box.line_width), y, length + 20))

    candidates.sort(key=lambda item: item[2], reverse=True)
    points: list[tuple[float, float]] = []
    radius = max(18.0, min(gray.shape[:2]) * 0.012)
    for x, y, _length in candidates:
        if x < 2 or y < 2 or x > gray.shape[1] - 2 or y > gray.shape[0] - 2:
            continue
        if any(((x - px) ** 2 + (y - py) ** 2) ** 0.5 <= radius for px, py in points):
            continue
        points.append((x, y))
        if len(points) >= max_points:
            break
    return points


def _build_label_montage(
    image_path: Path,
    boxes: list[LabelLineBox],
) -> tuple[Path, list[LabelMontagePlacement]]:
    source = Image.open(image_path).convert("RGB")
    max_width = 2200
    padding = 16
    crop_scale = 1.6
    x = padding
    y = padding
    row_height = 0
    crops: list[tuple[Image.Image, int, int]] = []
    placements: list[LabelMontagePlacement] = []

    for box in boxes:
        crop = source.crop(box.crop)
        crop = crop.resize(
            (
                max(1, int(crop.width * crop_scale)),
                max(1, int(crop.height * crop_scale)),
            ),
            Image.Resampling.LANCZOS,
        )
        if x + crop.width + padding > max_width:
            x = padding
            y += row_height + padding
            row_height = 0

        crops.append((crop, x, y))
        placements.append(
            LabelMontagePlacement(
                montage_x=x,
                montage_y=y,
                scale=crop_scale,
                source_box=box.crop,
                width=crop.width,
                height=crop.height,
            )
        )
        x += crop.width + padding
        row_height = max(row_height, crop.height)

    montage = Image.new("RGB", (max_width, y + row_height + padding), "white")
    for crop, crop_x, crop_y in crops:
        montage.paste(crop, (crop_x, crop_y))

    path = Path(tempfile.gettempdir()) / f"parts_label_montage_{os.getpid()}_{image_path.stem}.png"
    montage.save(path)
    return path, placements


def _build_endpoint_montage(
    image_path: Path,
    endpoints: list[tuple[float, float]],
) -> tuple[Path, list[LabelMontagePlacement]]:
    source = Image.open(image_path).convert("RGB")
    image_width, image_height = source.size
    crop_half = int(min(max(max(image_width, image_height) * 0.025, 70), 150))
    crop_scale = 2.0
    max_width = 2600
    padding = 14
    x = padding
    y = padding
    row_height = 0
    crops: list[tuple[Image.Image, int, int]] = []
    placements: list[LabelMontagePlacement] = []

    for point_x, point_y in endpoints:
        left = max(0, int(round(point_x - crop_half)))
        top = max(0, int(round(point_y - crop_half)))
        right = min(image_width, int(round(point_x + crop_half)))
        bottom = min(image_height, int(round(point_y + crop_half)))
        if right - left < 24 or bottom - top < 24:
            continue

        crop = source.crop((left, top, right, bottom)).resize(
            (
                max(1, int((right - left) * crop_scale)),
                max(1, int((bottom - top) * crop_scale)),
            ),
            Image.Resampling.LANCZOS,
        )
        if x + crop.width + padding > max_width:
            x = padding
            y += row_height + padding
            row_height = 0

        crops.append((crop, x, y))
        placements.append(
            LabelMontagePlacement(
                montage_x=x,
                montage_y=y,
                scale=crop_scale,
                source_box=(left, top, right, bottom),
                width=crop.width,
                height=crop.height,
            )
        )
        x += crop.width + padding
        row_height = max(row_height, crop.height)

    if not crops:
        path = Path(tempfile.gettempdir()) / f"parts_endpoint_montage_{os.getpid()}_{image_path.stem}.png"
        Image.new("RGB", (64, 64), "white").save(path)
        return path, []

    montage = Image.new("RGB", (max_width, y + row_height + padding), "white")
    for crop, crop_x, crop_y in crops:
        montage.paste(crop, (crop_x, crop_y))

    path = Path(tempfile.gettempdir()) / f"parts_endpoint_montage_{os.getpid()}_{image_path.stem}.png"
    montage.save(path)
    return path, placements


def _line_label_component_size_ok(width: int, height: int, area: int) -> bool:
    if height < 8 or height > 42 or width < 2 or width > 32:
        return False
    aspect = width / max(height, 1)
    if aspect < 0.05 or aspect > 1.20:
        return False
    density = area / max(width * height, 1)
    return 0.08 <= density <= 0.82


def _recognize_line_label_digit(binary_roi) -> tuple[str | None, float]:
    normalized = _normalize_binary_digit(binary_roi)
    if normalized is None:
        return None, 0.0

    best_digit: str | None = None
    best_score = -1.0
    for digit, template in _line_label_digit_templates():
        score = _binary_correlation(normalized, template)
        if score > best_score:
            best_digit = digit
            best_score = score

    threshold = 0.43 if best_digit == "1" else 0.36
    if best_digit is None or best_score < threshold:
        return None, best_score
    return best_digit, best_score


@lru_cache(maxsize=1)
def _line_label_digit_templates() -> tuple[tuple[str, object], ...]:
    templates = []
    for font_name in ("times.ttf", "timesbd.ttf", "arialbd.ttf"):
        for size in (20, 24, 28):
            try:
                font = ImageFont.truetype(font_name, size)
            except Exception:
                continue
            for digit in "0123456789":
                template = _render_digit_template(digit, font)
                if template is not None:
                    templates.append((digit, template))

    if not templates:
        return _digit_templates()
    return tuple(templates)


class OpenCVDigitBackend:
    name = "OpenCV digit detector"

    def available(self) -> bool:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except Exception:
            return False
        return True

    def recognize_digits(self, image_path: Path, known_numbers: set[str] | None = None) -> list[Hotspot]:
        if not self.available():
            raise OcrError("OpenCV не установлен")

        import cv2
        import numpy as np

        with Image.open(image_path) as image:
            gray = np.array(image.convert("L"))

        # Work on dark foreground. A small opening removes isolated scan noise
        # while keeping printed digits mostly intact.
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
        image_height, image_width = gray.shape[:2]
        candidates: list[dict[str, float | str]] = []

        for label in range(1, count):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if not self._component_size_ok(w, h, area, image_width, image_height):
                continue

            roi = binary[y : y + h, x : x + w]
            digit, score = self._recognize_component(roi)
            if digit is None:
                continue

            candidates.append(
                {
                    "digit": digit,
                    "score": score,
                    "x": float(x),
                    "y": float(y),
                    "w": float(w),
                    "h": float(h),
                    "cx": float(x + w / 2),
                    "cy": float(y + h / 2),
                }
            )

        groups = self._group_digits(candidates)
        spots: list[Hotspot] = []
        for group in groups:
            digits = "".join(str(item["digit"]) for item in group)
            number = normalize_number(digits)
            if known_numbers and number not in known_numbers:
                continue

            left = min(float(item["x"]) for item in group)
            top = min(float(item["y"]) for item in group)
            right = max(float(item["x"]) + float(item["w"]) for item in group)
            bottom = max(float(item["y"]) + float(item["h"]) for item in group)
            avg_score = sum(float(item["score"]) for item in group) / len(group)
            spots.append(
                Hotspot(
                    number=number,
                    x=(left + right) / 2,
                    y=(top + bottom) / 2,
                    width=max(right - left, 1),
                    height=max(bottom - top, 1),
                    source=f"{self.name} {avg_score:.2f}",
                )
            )

        return _dedupe_spots(spots)

    def _component_size_ok(self, width: int, height: int, area: int, image_width: int, image_height: int) -> bool:
        if height < 7 or width < 3:
            return False
        if height > image_height * 0.18 or width > image_width * 0.12:
            return False
        aspect = width / max(height, 1)
        if aspect < 0.08 or aspect > 1.15:
            return False
        density = area / max(width * height, 1)
        if density < 0.12 or density > 0.86:
            return False
        if width <= 3 and height > 28:
            return False
        return True

    def _remove_long_lines(self, binary):
        import cv2
        import numpy as np

        height, width = binary.shape[:2]
        min_length = max(35, int(min(width, height) * 0.10))
        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=max(25, min_length // 2),
            minLineLength=min_length,
            maxLineGap=6,
        )
        if lines is None:
            return binary

        cleaned = binary.copy()
        for line in lines[:, 0]:
            x1, y1, x2, y2 = [int(value) for value in line]
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if length < min_length:
                continue
            cv2.line(cleaned, (x1, y1), (x2, y2), 0, thickness=3)
        return cleaned

    def _recognize_component(self, roi) -> tuple[str | None, float]:
        import numpy as np

        normalized = _normalize_binary_digit(roi)
        if normalized is None:
            return None, 0.0

        best_digit: str | None = None
        best_score = -1.0
        for digit, template in _digit_templates():
            score = _binary_correlation(normalized, template)
            if score > best_score:
                best_digit = digit
                best_score = score

        threshold = 0.48 if best_digit == "1" else 0.42
        if best_digit is None or best_score < threshold:
            return None, best_score

        foreground_ratio = float(np.mean(normalized > 0))
        if best_digit == "1" and foreground_ratio < 0.035:
            return None, best_score
        return best_digit, best_score

    def _group_digits(self, candidates: list[dict[str, float | str]]) -> list[list[dict[str, float | str]]]:
        candidates = sorted(candidates, key=lambda item: (float(item["cy"]), float(item["x"])))
        used: set[int] = set()
        groups: list[list[dict[str, float | str]]] = []

        for index, candidate in enumerate(candidates):
            if index in used:
                continue
            group = [candidate]
            used.add(index)

            changed = True
            while changed:
                changed = False
                group_right = max(float(item["x"]) + float(item["w"]) for item in group)
                group_height = max(float(item["h"]) for item in group)
                group_cy = sum(float(item["cy"]) for item in group) / len(group)

                for other_index, other in enumerate(candidates):
                    if other_index in used:
                        continue
                    horizontal_gap = float(other["x"]) - group_right
                    vertical_delta = abs(float(other["cy"]) - group_cy)
                    max_gap = max(group_height * 0.75, 10)
                    if 0 <= horizontal_gap <= max_gap and vertical_delta <= group_height * 0.45:
                        group.append(other)
                        used.add(other_index)
                        changed = True

            groups.append(sorted(group, key=lambda item: float(item["x"])))

        return groups


class MultiScaleOpenCVDigitBackend:
    name = "OpenCV multiscale detector"

    def available(self) -> bool:
        return OpenCVDigitBackend().available()

    def recognize_digits(self, image_path: Path, known_numbers: set[str] | None = None) -> list[Hotspot]:
        if not self.available():
            raise OcrError("OpenCV не установлен")

        backend = OpenCVDigitBackend()
        spots: list[Hotspot] = []
        with Image.open(image_path) as image:
            source = image.convert("RGB")
            source_width, source_height = source.size
            for target_size in OPENCV_MULTISCALE_SIZES:
                scale = target_size / max(source_width, source_height, 1)
                target_width = max(1, int(round(source_width * scale)))
                target_height = max(1, int(round(source_height * scale)))
                scaled = source.resize((target_width, target_height), Image.Resampling.LANCZOS)
                scaled_path = Path(tempfile.gettempdir()) / (
                    f"parts_opencv_multiscale_{os.getpid()}_{image_path.stem}_{target_size}.png"
                )
                scaled.save(scaled_path)
                try:
                    scale_spots = backend.recognize_digits(scaled_path, known_numbers)
                finally:
                    scaled_path.unlink(missing_ok=True)

                for spot in scale_spots:
                    spot.x /= scale
                    spot.y /= scale
                    spot.width /= scale
                    spot.height /= scale
                    spot.source = f"{spot.source}; multiscale {target_size}"
                spots.extend(scale_spots)

        return _dedupe_spots(spots)


def _normalize_binary_digit(binary_roi):
    import cv2
    import numpy as np

    ys, xs = np.where(binary_roi > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    cropped = binary_roi[min(ys) : max(ys) + 1, min(xs) : max(xs) + 1]
    height, width = cropped.shape[:2]
    if height <= 0 or width <= 0:
        return None

    canvas_width, canvas_height = 32, 48
    scale = min((canvas_width - 8) / width, (canvas_height - 8) / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    x = (canvas_width - new_width) // 2
    y = (canvas_height - new_height) // 2
    canvas[y : y + new_height, x : x + new_width] = resized
    _, canvas = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
    return canvas


@lru_cache(maxsize=1)
def _digit_templates() -> tuple[tuple[str, object], ...]:
    templates = []
    font_names = [
        "arial.ttf",
        "arialbd.ttf",
        "calibri.ttf",
        "calibrib.ttf",
        "segoeui.ttf",
        "segoeuib.ttf",
        "tahoma.ttf",
        "times.ttf",
        "timesbd.ttf",
    ]
    font_sizes = [26, 32, 38, 46, 56, 66]
    for font_name in font_names:
        for size in font_sizes:
            try:
                font = ImageFont.truetype(font_name, size)
            except Exception:
                continue
            for digit in "0123456789":
                template = _render_digit_template(digit, font)
                if template is not None:
                    templates.append((digit, template))

    if not templates:
        font = ImageFont.load_default()
        for digit in "0123456789":
            template = _render_digit_template(digit, font)
            if template is not None:
                templates.append((digit, template))

    return tuple(templates)


def _render_digit_template(digit: str, font: ImageFont.ImageFont):
    import numpy as np

    image = Image.new("L", (96, 128), 0)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), digit, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.text(
        ((96 - text_width) / 2 - bbox[0], (128 - text_height) / 2 - bbox[1]),
        digit,
        fill=255,
        font=font,
    )
    array = np.array(image)
    _, binary = __import__("cv2").threshold(array, 32, 255, __import__("cv2").THRESH_BINARY)
    return _normalize_binary_digit(binary)


def _binary_correlation(a, b) -> float:
    import numpy as np

    a_float = (a > 0).astype("float32").ravel()
    b_float = (b > 0).astype("float32").ravel()
    a_float -= a_float.mean()
    b_float -= b_float.mean()
    denominator = float(np.linalg.norm(a_float) * np.linalg.norm(b_float))
    if denominator <= 1e-6:
        return -1.0
    return float(np.dot(a_float, b_float) / denominator)


def _dedupe_spots(spots: list[Hotspot]) -> list[Hotspot]:
    result: list[Hotspot] = []
    for spot in spots:
        duplicate = False
        for existing in result:
            if (
                existing.number == spot.number
                and abs(existing.x - spot.x) < max(existing.width, spot.width, 12)
                and abs(existing.y - spot.y) < max(existing.height, spot.height, 12)
            ):
                duplicate = True
                break
        if not duplicate:
            result.append(spot)
    return sorted(result, key=lambda item: (int(item.number), item.y, item.x))


def parse_articles(text: str) -> dict[str, str]:
    articles: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        number_matches = list(NUMBER_RE.finditer(line))
        numbers_only = bool(number_matches) and all(
            char.isdigit() or char.isspace() or char in ",;:./\\|-_()[]{}"
            for char in line
        )
        if numbers_only:
            line_numbers = [
                normalize_number(token)
                for token in re.split(r"[\s,;:./\\|\-_()\[\]{}]+", line)
                if token
            ]
            line_is_plain_number_list = all(
                number.isdigit() and int(number) <= 500
                for number in line_numbers
            )
            if line_is_plain_number_list:
                for number in line_numbers:
                    articles.setdefault(number, "")
                continue

        match = ARTICLE_LINE_RE.match(line)
        if match:
            number = normalize_number(match.group(1))
            if number.isdigit() and int(number) > 500:
                continue
            article = match.group(2).strip()
        else:
            number_matches = list(NUMBER_RE.finditer(line))
            if not number_matches:
                continue
            number_match = number_matches[0]
            first_number = normalize_number(number_match.group(1))
            if first_number.isdigit() and int(first_number) > 500:
                number_match = None
                for candidate in reversed(number_matches[1:]):
                    candidate_number = normalize_number(candidate.group(1))
                    if not candidate_number.isdigit() or int(candidate_number) > 500:
                        continue
                    previous_char = line[candidate.start() - 1] if candidate.start() > 0 else ""
                    if previous_char in "-_№# ":
                        number_match = candidate
                        break
                if number_match is None:
                    continue
            number = normalize_number(number_match.group(1))
            if number.isdigit() and int(number) > 500:
                continue
            before = line[: number_match.start()].strip()
            after = line[number_match.end() :].strip()
            if before and any(char.isalnum() for char in before):
                article = line
            else:
                article = after
            article = article.strip(" \t.;,:-–—)")

        if not article:
            continue

        articles[number] = article
    return articles

class PartsHotspotApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"Привязка артикулов к цифрам на PDF - build {APP_BUILD}")
        self.geometry("1500x900")
        self.minsize(980, 640)

        self.image_path: Path | None = None
        self.source_path: Path | None = None
        self.source_type = "image"
        self.original_image: Image.Image | None = None
        self.raster_source_image: Image.Image | None = None
        self.preview_image: Image.Image | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.minimap_photo: ImageTk.PhotoImage | None = None
        self.preview_scale = 1.0
        self.preview_offset_x = 0
        self.preview_offset_y = 0
        self.source_image_size: tuple[int, int] = (WORK_IMAGE_SIZE, WORK_IMAGE_SIZE)
        self.source_to_work_scale = 1.0
        self.source_to_work_offset_x = 0.0
        self.source_to_work_offset_y = 0.0
        self.pdf_source_to_work_scale = 1.0
        self.pdf_source_to_work_offset_x = 0.0
        self.pdf_source_to_work_offset_y = 0.0
        self.viewport_zoom = 1.0
        self.viewport_pan_x = 0.0
        self.viewport_pan_y = 0.0
        self.panning = False
        self.pan_start: tuple[int, int] | None = None
        self.pan_origin: tuple[float, float] | None = None
        self.leader_line_cache_key: tuple[str, tuple[int, int]] | None = None
        self.leader_line_cache: list[LeaderLineSegment] = []
        self.line_score_cache: dict[tuple[float, float, float, float, str], float] = {}
        self.brush_strokes: list[tuple[float, float, float, float, float]] = []
        self.cutout_regions: list[CutoutRegion] = []
        self.image_before_brush: Image.Image | None = None
        self.brush_mode = tk.BooleanVar(value=False)
        self.brush_size = tk.DoubleVar(value=18.0)
        self.brush_dragging = False
        self.brush_last_point: tuple[float, float] | None = None
        self.sidebar_width = 255
        self.sidebar_visible = True
        self.sidebar_toggle_text = tk.StringVar(value="Скрыть панель")
        self.fragments: list[ImageFragment] = []
        self.active_fragment_index: int | None = None
        self.fragment_buttons: list[ttk.Button] = []
        self.fragment_button_frame: ttk.Frame | None = None
        self.fragment_count_var = tk.StringVar(value="Фрагменты: 0")
        self.fragment_search_running = False
        self.fragment_progress_var = tk.DoubleVar(value=0.0)
        self.fragment_progress_text = tk.StringVar(value="Поиск не запущен")
        self.ui_queue = queue.Queue()
        self.spot_filter = tk.StringVar(value="Все")
        self.problem_mode = tk.BooleanVar(value=False)
        self.temp_image_paths: list[Path] = []

        self.pdf_document = None
        self.pdf_page_index = 0
        self.pdf_page_count = 0
        self.pdf_render_scale = 2.0
        self.pdf_ocr_render_scale = 5.5
        self.current_image_is_pdf_page = False
        self.pdf_page_status = tk.StringVar(value="PDF не открыт")

        self.articles: dict[str, str] = {}
        self.spots: list[Hotspot] = []
        self.candidate_spots_by_number: dict[str, list[Hotspot]] = {}
        self.current_candidate_options: list[Hotspot] = []
        self.candidate_var = tk.StringVar(value="")
        self.ocr_run_id = 0
        self.selected_index: int | None = None
        self.dragging_index: int | None = None
        self.add_mode = tk.BooleanVar(value=False)
        self.crop_mode = tk.BooleanVar(value=False)
        self.lasso_mode = tk.BooleanVar(value=False)
        self.crop_all_pdf_pages = tk.BooleanVar(value=True)
        self.rotation_angle_var = tk.StringVar(value="0")
        self.quality_auto_trim_var = tk.BooleanVar(value=True)
        self.quality_drawing_mode_var = tk.BooleanVar(value=False)
        self.quality_line_sharpen_var = tk.BooleanVar(value=False)
        self.quality_redraw_numbers_var = tk.BooleanVar(value=False)
        self.quality_supersample_var = tk.StringVar(value="3200")
        self.crop_rect: tuple[float, float, float, float] | None = None
        self.crop_start: tuple[float, float] | None = None
        self.crop_dragging = False
        self.lasso_points: list[tuple[float, float]] = []
        self.lasso_dragging = False
        self.status = tk.StringVar(
            value=f"Build {APP_BUILD}. Откройте PDF или изображение и вставьте список номеров с артикулами."
        )
        self.ocr_status = tk.StringVar(value=self._backend_status_text())

        self._build_ui()
        self._bind_events()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, self.process_ui_queue)

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.configure("TButton", padding=(4, 2))
        style.configure("TCheckbutton", padding=(1, 1))
        style.configure("TLabelframe", padding=4)
        style.configure("Treeview", rowheight=20)

        root = ttk.Frame(self, padding=3)
        root.pack(fill=tk.BOTH, expand=True)
        self.root_frame = root

        root.columnconfigure(0, minsize=self.sidebar_width, weight=0)
        root.columnconfigure(1, weight=1)
        root.columnconfigure(2, minsize=150, weight=0)
        root.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(root, width=self.sidebar_width)
        self.sidebar = sidebar
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(7, weight=1)
        sidebar.rowconfigure(11, weight=1)

        image_buttons = ttk.Frame(sidebar)
        image_buttons.grid(row=0, column=0, sticky="ew")
        image_buttons.columnconfigure(0, weight=1)

        ttk.Button(image_buttons, text="Открыть файл", command=self.open_file).grid(
            row=0, column=0, sticky="ew"
        )

        ttk.Label(sidebar, textvariable=self.ocr_status, foreground="#555").grid(
            row=1, column=0, sticky="w", pady=(4, 7)
        )

        pdf_buttons = ttk.Frame(sidebar)
        pdf_buttons.grid(row=2, column=0, sticky="ew", pady=(0, 5))
        pdf_buttons.columnconfigure(1, weight=1)
        ttk.Button(pdf_buttons, text="← PDF", command=self.prev_pdf_page).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Label(pdf_buttons, textvariable=self.pdf_page_status, anchor="center").grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(pdf_buttons, text="PDF →", command=self.next_pdf_page).grid(
            row=0, column=2, sticky="ew", padx=(4, 0)
        )

        crop_box = ttk.LabelFrame(sidebar, text="Обрезка", padding=4)
        crop_box.grid(row=3, column=0, sticky="ew", pady=(0, 5))
        crop_box.columnconfigure(0, weight=1)
        crop_box.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            crop_box,
            text="Выделять область мышью",
            variable=self.crop_mode,
            command=self.toggle_crop_mode,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(
            crop_box,
            text="Лассо: свободная область",
            variable=self.lasso_mode,
            command=self.toggle_lasso_mode,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))
        ttk.Checkbutton(
            crop_box,
            text="PDF: применить ко всем страницам",
            variable=self.crop_all_pdf_pages,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))
        ttk.Button(crop_box, text="Применить", command=self.apply_crop).grid(
            row=3, column=0, sticky="ew", padx=(0, 3), pady=(5, 0)
        )
        ttk.Button(crop_box, text="Сбросить", command=self.clear_crop).grid(
            row=3, column=1, sticky="ew", padx=(3, 0), pady=(5, 0)
        )
        ttk.Button(crop_box, text="Сохранить обрезанное", command=self.save_cropped_file).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )
        ttk.Button(crop_box, text="PNG HD", command=self.save_high_quality_png).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(4, 0)
        )
        ttk.Button(crop_box, text="Добавить фрагмент справа", command=self.add_fragment_from_selection).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(4, 0)
        )
        rotate_buttons = ttk.Frame(crop_box)
        rotate_buttons.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        rotate_buttons.columnconfigure(0, weight=1)
        rotate_buttons.columnconfigure(1, weight=1)
        ttk.Button(rotate_buttons, text="90° влево", command=self.rotate_left_90).grid(
            row=0, column=0, sticky="ew", padx=(0, 3)
        )
        ttk.Button(rotate_buttons, text="90° вправо", command=self.rotate_right_90).grid(
            row=0, column=1, sticky="ew", padx=(3, 0)
        )
        rotate_manual = ttk.Frame(crop_box)
        rotate_manual.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        rotate_manual.columnconfigure(1, weight=1)
        ttk.Label(rotate_manual, text="Угол").grid(row=0, column=0, sticky="w")
        ttk.Entry(rotate_manual, textvariable=self.rotation_angle_var, width=7).grid(
            row=0, column=1, sticky="ew", padx=(5, 4)
        )
        ttk.Button(rotate_manual, text="Повернуть", command=self.rotate_by_custom_angle).grid(
            row=0, column=2, sticky="ew"
        )

        quality_box = ttk.LabelFrame(sidebar, text="Детализация 800x800", padding=4)
        quality_box.grid(row=4, column=0, sticky="ew", pady=(0, 5))
        quality_box.columnconfigure(0, weight=1)
        quality_box.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            quality_box,
            text="Автополя",
            variable=self.quality_auto_trim_var,
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            quality_box,
            text="Чертёж",
            variable=self.quality_drawing_mode_var,
        ).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(
            quality_box,
            text="Резкость линий",
            variable=self.quality_line_sharpen_var,
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Checkbutton(
            quality_box,
            text="Перерисовать №",
            variable=self.quality_redraw_numbers_var,
        ).grid(row=1, column=1, sticky="w", pady=(2, 0))
        supersample_row = ttk.Frame(quality_box)
        supersample_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        supersample_row.columnconfigure(1, weight=1)
        ttk.Label(supersample_row, text="Сборка").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            supersample_row,
            textvariable=self.quality_supersample_var,
            values=("2400", "3200", "4000"),
            width=7,
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", padx=(5, 0))

        brush_box = ttk.LabelFrame(sidebar, text="Кисть", padding=4)
        brush_box.grid(row=5, column=0, sticky="ew", pady=(0, 5))
        brush_box.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            brush_box,
            text="Убирать лишнее",
            variable=self.brush_mode,
            command=self.toggle_brush_mode,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(brush_box, text="Размер").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Scale(
            brush_box,
            from_=6,
            to=64,
            variable=self.brush_size,
            orient=tk.HORIZONTAL,
        ).grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=(4, 0))
        ttk.Button(brush_box, text="Сбросить кисть", command=self.clear_brush_edits).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )

        ttk.Label(sidebar, text="Список номеров и артикулов").grid(row=6, column=0, sticky="w")
        self.articles_text = tk.Text(sidebar, height=6, wrap="word", undo=True)
        self.articles_text.grid(row=7, column=0, sticky="nsew", pady=(2, 4))

        article_buttons = ttk.Frame(sidebar)
        article_buttons.grid(row=8, column=0, sticky="ew", pady=(0, 5))
        article_buttons.columnconfigure(0, weight=1)
        article_buttons.columnconfigure(1, weight=1)
        ttk.Button(article_buttons, text="Найти по списку", command=self.bind_articles_to_digits).grid(
            row=0, column=0, columnspan=2, sticky="ew"
        )
        ttk.Button(article_buttons, text="Очистить точки", command=self.clear_spots).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0)
        )

        ttk.Label(sidebar, text="Найденные/добавленные цифры").grid(row=9, column=0, sticky="w")
        columns = ("number", "x", "y", "article")
        self.tree = ttk.Treeview(sidebar, columns=columns, show="headings", height=5)
        self.tree.heading("number", text="№")
        self.tree.heading("x", text="X%")
        self.tree.heading("y", text="Y%")
        self.tree.heading("article", text="Артикул")
        self.tree.column("number", width=36, anchor=tk.CENTER, stretch=False)
        self.tree.column("x", width=46, anchor=tk.E, stretch=False)
        self.tree.column("y", width=46, anchor=tk.E, stretch=False)
        self.tree.column("article", width=105, anchor=tk.W)
        self.tree.grid(row=10, column=0, sticky="nsew", pady=(2, 4))

        editor = ttk.LabelFrame(sidebar, text="Правка выбранной цифры", padding=4)
        editor.grid(row=11, column=0, sticky="nsew")
        editor.columnconfigure(1, weight=1)

        ttk.Label(editor, text="Номер").grid(row=0, column=0, sticky="w")
        self.number_entry = ttk.Entry(editor, width=12)
        self.number_entry.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=1)

        ttk.Label(editor, text="Артикул").grid(row=1, column=0, sticky="w")
        self.article_entry = ttk.Entry(editor)
        self.article_entry.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=1)

        edit_buttons = ttk.Frame(editor)
        edit_buttons.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        edit_buttons.columnconfigure(0, weight=1)
        edit_buttons.columnconfigure(1, weight=1)
        ttk.Button(edit_buttons, text="Сохранить", command=self.apply_editor).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(edit_buttons, text="Удалить", command=self.delete_selected).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )

        ttk.Checkbutton(
            editor,
            text="Добавлять цифру кликом по картинке",
            variable=self.add_mode,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Label(editor, text="Кандидаты").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.candidate_combo = ttk.Combobox(
            editor,
            textvariable=self.candidate_var,
            state="readonly",
            values=(),
        )
        self.candidate_combo.grid(row=4, column=1, sticky="ew", padx=(6, 0), pady=(6, 0))
        ttk.Button(editor, text="Поставить выбранный кандидат", command=self.apply_selected_candidate).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )

        export_buttons = ttk.Frame(sidebar)
        export_buttons.grid(row=12, column=0, sticky="ew", pady=(5, 0))
        export_buttons.columnconfigure(0, weight=1)
        export_buttons.columnconfigure(1, weight=1)
        ttk.Button(export_buttons, text="Экспорт CSV", command=self.export_csv).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(export_buttons, text="Экспорт JSON", command=self.export_json).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )

        canvas_frame = ttk.Frame(root)
        canvas_frame.grid(row=0, column=1, sticky="nsew")
        canvas_frame.rowconfigure(1, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        view_tools = ttk.Frame(canvas_frame)
        view_tools.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        view_tools.columnconfigure(1, weight=1)
        ttk.Label(view_tools, text="Фильтр").grid(row=0, column=0, sticky="w")
        self.filter_combo = ttk.Combobox(
            view_tools,
            textvariable=self.spot_filter,
            state="readonly",
            values=(
                "Все",
                "Проблемные",
                "Розовые",
                "Вычисленные",
                "Жёлтые",
                "Без артикула",
                "С линией",
            ),
            width=16,
        )
        self.filter_combo.grid(row=0, column=1, sticky="w", padx=(8, 12))
        self.filter_combo.bind("<<ComboboxSelected>>", self.on_filter_changed)
        ttk.Checkbutton(
            view_tools,
            text="Только проблемные",
            variable=self.problem_mode,
            command=self.toggle_problem_mode,
        ).grid(row=0, column=2, sticky="w", padx=(0, 8))
        ttk.Button(view_tools, text="←", width=4, command=self.prev_problem_spot).grid(
            row=0, column=3, sticky="e", padx=(0, 4)
        )
        ttk.Button(view_tools, text="→", width=4, command=self.next_problem_spot).grid(
            row=0, column=4, sticky="e"
        )
        ttk.Button(
            view_tools,
            textvariable=self.sidebar_toggle_text,
            command=self.toggle_sidebar,
            width=14,
        ).grid(row=0, column=5, sticky="e", padx=(8, 0))

        self.canvas = tk.Canvas(canvas_frame, background="#f3f5f7", highlightthickness=1, highlightbackground="#ccd2d8")
        self.canvas.grid(row=1, column=0, sticky="nsew")

        self.minimap = tk.Canvas(
            canvas_frame,
            height=44,
            background="#f8fafc",
            highlightthickness=1,
            highlightbackground="#ccd2d8",
        )
        self.minimap.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        self.minimap.bind("<Button-1>", self.on_minimap_click)

        fragments_panel = ttk.LabelFrame(root, text="Фрагменты", padding=4)
        self.fragments_panel = fragments_panel
        fragments_panel.grid(row=0, column=2, sticky="nsew", padx=(4, 0))
        fragments_panel.columnconfigure(0, weight=1)
        fragments_panel.rowconfigure(3, weight=1)

        ttk.Label(fragments_panel, textvariable=self.fragment_count_var, anchor="center").grid(
            row=0, column=0, sticky="ew", pady=(0, 4)
        )
        self.fragment_progress = ttk.Progressbar(
            fragments_panel,
            variable=self.fragment_progress_var,
            maximum=100,
            mode="determinate",
        )
        self.fragment_progress.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 3))
        ttk.Label(
            fragments_panel,
            textvariable=self.fragment_progress_text,
            anchor="center",
            wraplength=132,
        ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self.fragment_canvas = tk.Canvas(
            fragments_panel,
            width=132,
            background="#f8fafc",
            highlightthickness=1,
            highlightbackground="#ccd2d8",
        )
        self.fragment_canvas.grid(row=3, column=0, sticky="nsew")
        self.fragment_scrollbar = ttk.Scrollbar(
            fragments_panel,
            orient=tk.VERTICAL,
            command=self.fragment_canvas.yview,
        )
        self.fragment_scrollbar.grid(row=3, column=1, sticky="ns")
        self.fragment_canvas.configure(yscrollcommand=self.fragment_scrollbar.set)
        self.fragment_button_frame = ttk.Frame(self.fragment_canvas)
        self.fragment_window = self.fragment_canvas.create_window(
            (0, 0),
            window=self.fragment_button_frame,
            anchor=tk.NW,
        )
        self.fragment_button_frame.bind("<Configure>", self.on_fragment_frame_configure)
        self.fragment_canvas.bind("<Configure>", self.on_fragment_canvas_configure)
        ttk.Button(fragments_panel, text="Удалить фрагмент", command=self.delete_current_fragment).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )

        ttk.Label(root, textvariable=self.status, anchor="w").grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=(2, 0)
        )

    def toggle_sidebar(self) -> None:
        if self.sidebar_visible:
            self.sidebar.grid_remove()
            self.root_frame.columnconfigure(0, minsize=0)
            self.sidebar_toggle_text.set("Показать панель")
            self.sidebar_visible = False
        else:
            self.root_frame.columnconfigure(0, minsize=self.sidebar_width)
            self.sidebar.grid()
            self.sidebar_toggle_text.set("Скрыть панель")
            self.sidebar_visible = True
        self.after_idle(self.redraw_canvas)

    def _bind_events(self) -> None:
        self.canvas.bind("<Configure>", lambda _event: self.redraw_canvas())
        self.canvas.bind("<Button-1>", self.on_canvas_down)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_up)
        self.canvas.bind("<MouseWheel>", self.on_canvas_wheel)
        self.canvas.bind("<Button-4>", self.on_canvas_wheel)
        self.canvas.bind("<Button-5>", self.on_canvas_wheel)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.bind_text_paste_shortcuts(self.articles_text)
        self.bind_text_paste_shortcuts(self.number_entry)
        self.bind_text_paste_shortcuts(self.article_entry)

    def bind_text_paste_shortcuts(self, widget: tk.Widget) -> None:
        widget.bind("<Control-KeyPress>", self.on_text_control_keypress)
        widget.bind("<Shift-Insert>", self.paste_from_clipboard)

    def on_text_control_keypress(self, event: tk.Event) -> str | None:
        key = str(getattr(event, "keysym", "")).lower()
        char = str(getattr(event, "char", ""))
        keycode = int(getattr(event, "keycode", 0) or 0)
        paste_keys = {"v", "cyrillic_em"}
        if key in paste_keys or char in {"\x16", "м", "М"} or keycode == 86:
            return self.paste_from_clipboard(event)
        return None

    def paste_from_clipboard(self, event: tk.Event) -> str:
        widget = event.widget
        if not isinstance(widget, (tk.Text, tk.Entry, ttk.Entry, ttk.Combobox)):
            return "break"
        try:
            text = self.clipboard_get()
        except tk.TclError:
            return "break"

        try:
            widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            pass
        widget.insert(tk.INSERT, text)
        return "break"

    def _backend_status_text(self) -> str:
        available = [backend.name for backend in self._ocr_backends() if backend.available()]
        if available:
            return "OCR доступен: " + ", ".join(available)
        return "OCR пока недоступен: установите зависимости через run.bat"

    def _ocr_backends(
        self,
    ) -> list[
        WindowsOcrBackend
        | LineLabelWindowsBackend
        | LineLabelDigitBackend
        | LeaderEndpointOcrBackend
        | DenseCropWindowsOcrBackend
        | GridTileOcrBackend
        | CircledNumberBackend
        | OpenCVDigitBackend
        | MultiScaleOpenCVDigitBackend
        | TesseractBackend
    ]:
        if self.current_image_is_pdf_page:
            return [
                GridTileOcrBackend(),
                WindowsOcrBackend(),
                LineLabelWindowsBackend(),
                LeaderEndpointOcrBackend(),
                LineLabelDigitBackend(),
                DenseCropWindowsOcrBackend(),
                OpenCVDigitBackend(),
                TesseractBackend(),
            ]
        return [
            GridTileOcrBackend(),
            WindowsOcrBackend(),
            LeaderEndpointOcrBackend(),
            CircledNumberBackend(),
            OpenCVDigitBackend(),
            MultiScaleOpenCVDigitBackend(),
            TesseractBackend(),
        ]

    def get_fitz(self, show_error: bool = True):
        try:
            import fitz
        except Exception as exc:
            if show_error:
                messagebox.showerror(
                    "PDF недоступен",
                    "Для работы с PDF нужен пакет PyMuPDF.\n"
                    "Запустите run.bat, чтобы установить зависимости.\n\n"
                    f"{exc}",
                )
            return None
        return fitz

    def write_temp_image(self, image: Image.Image, label: str) -> Path:
        safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", label).strip("_") or "image"
        path = Path(tempfile.gettempdir()) / f"parts_markup_{os.getpid()}_{safe_label}.png"
        image.save(path)
        if path not in self.temp_image_paths:
            self.temp_image_paths.append(path)
        return path

    def close_pdf_document(self) -> None:
        if self.pdf_document is not None:
            try:
                self.pdf_document.close()
            except Exception:
                pass
        self.pdf_document = None

    def cleanup_temp_files(self) -> None:
        for path in self.temp_image_paths:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
        self.temp_image_paths.clear()

    def on_close(self) -> None:
        self.close_pdf_document()
        self.cleanup_temp_files()
        self.destroy()

    def enqueue_ui(self, callback) -> None:
        self.ui_queue.put(callback)

    def process_ui_queue(self) -> None:
        try:
            while True:
                callback = self.ui_queue.get_nowait()
                try:
                    callback()
                except Exception as exc:
                    try:
                        self.status.set(f"Ошибка обновления UI: {exc}")
                    except Exception:
                        pass
        except queue.Empty:
            pass
        try:
            if self.winfo_exists():
                self.after(100, self.process_ui_queue)
        except tk.TclError:
            pass

    def reset_viewport(self) -> None:
        self.viewport_zoom = 1.0
        self.viewport_pan_x = 0.0
        self.viewport_pan_y = 0.0
        self.panning = False
        self.pan_start = None
        self.pan_origin = None

    def reset_brush_state(self) -> None:
        self.brush_strokes = []
        self.image_before_brush = None
        self.brush_dragging = False
        self.brush_last_point = None
        self.leader_line_cache_key = None
        self.leader_line_cache = []
        self.line_score_cache = {}

    def set_work_image_from_source(self, image: Image.Image, label: str) -> None:
        normalized, scale, offset_x, offset_y = normalize_to_work_square(image)
        self.source_image_size = image.size
        self.source_to_work_scale = scale
        self.source_to_work_offset_x = offset_x
        self.source_to_work_offset_y = offset_y
        self.original_image = normalized
        self.image_path = self.write_temp_image(normalized, label)
        self.leader_line_cache_key = None
        self.leader_line_cache = []
        self.line_score_cache = {}

    def clone_hotspots(self, spots: list[Hotspot]) -> list[Hotspot]:
        return [Hotspot(**asdict(spot)) for spot in spots]

    def clone_cutout_regions(self, regions: list[CutoutRegion]) -> list[CutoutRegion]:
        return [
            CutoutRegion(
                box=region.box,
                mask=region.mask.copy() if region.mask is not None else None,
            )
            for region in regions
        ]

    def current_fragment_snapshot(self, name: str, label: str) -> ImageFragment:
        assert self.original_image is not None
        image = self.original_image.copy()
        return ImageFragment(
            name=name,
            image=image,
            image_path=self.write_temp_image(image, label),
            source_image_size=self.source_image_size,
            source_to_work_scale=self.source_to_work_scale,
            source_to_work_offset_x=self.source_to_work_offset_x,
            source_to_work_offset_y=self.source_to_work_offset_y,
            pdf_source_to_work_scale=self.pdf_source_to_work_scale,
            pdf_source_to_work_offset_x=self.pdf_source_to_work_offset_x,
            pdf_source_to_work_offset_y=self.pdf_source_to_work_offset_y,
            current_image_is_pdf_page=self.current_image_is_pdf_page,
            raster_source_image=self.raster_source_image.copy() if self.raster_source_image is not None else None,
            spots=self.clone_hotspots(self.spots),
            brush_strokes=list(self.brush_strokes),
            cutout_regions=self.clone_cutout_regions(self.cutout_regions),
        )

    def reset_fragments_from_current(self, name: str = "Исходник") -> None:
        self.fragments = []
        self.active_fragment_index = None
        if self.original_image is not None:
            self.fragments.append(self.current_fragment_snapshot(name, "fragment_original_800"))
            self.active_fragment_index = 0
        self.refresh_fragment_buttons()

    def save_current_fragment_state(self, refresh: bool = False) -> None:
        if self.original_image is None or self.active_fragment_index is None:
            return
        if not (0 <= self.active_fragment_index < len(self.fragments)):
            return

        name = self.fragments[self.active_fragment_index].name
        label = f"fragment_{self.active_fragment_index + 1}_800"
        self.fragments[self.active_fragment_index] = self.current_fragment_snapshot(name, label)
        if refresh:
            self.refresh_fragment_buttons()

    def load_fragment_state(self, fragment: ImageFragment, update_ui: bool = True) -> None:
        self.original_image = fragment.image.copy()
        self.image_path = fragment.image_path
        self.source_image_size = fragment.source_image_size
        self.source_to_work_scale = fragment.source_to_work_scale
        self.source_to_work_offset_x = fragment.source_to_work_offset_x
        self.source_to_work_offset_y = fragment.source_to_work_offset_y
        self.pdf_source_to_work_scale = fragment.pdf_source_to_work_scale
        self.pdf_source_to_work_offset_x = fragment.pdf_source_to_work_offset_x
        self.pdf_source_to_work_offset_y = fragment.pdf_source_to_work_offset_y
        self.current_image_is_pdf_page = fragment.current_image_is_pdf_page
        self.raster_source_image = fragment.raster_source_image.copy() if fragment.raster_source_image is not None else None
        self.spots = self.clone_hotspots(fragment.spots)
        self.brush_strokes = list(fragment.brush_strokes)
        self.cutout_regions = self.clone_cutout_regions(fragment.cutout_regions)
        self.image_before_brush = None
        self.brush_dragging = False
        self.brush_last_point = None
        self.preview_image = None
        self.preview_photo = None
        self.crop_rect = None
        self.crop_start = None
        self.crop_dragging = False
        self.lasso_points = []
        self.lasso_dragging = False
        self.selected_index = None
        self.candidate_spots_by_number = {}
        self.current_candidate_options = []
        if update_ui:
            self.candidate_var.set("")
        self.ocr_run_id += 1
        self.leader_line_cache_key = None
        self.leader_line_cache = []
        self.line_score_cache = {}
        self.reset_viewport()

    def switch_fragment(self, index: int, save_current: bool = True) -> None:
        if not (0 <= index < len(self.fragments)):
            return
        if save_current and index != self.active_fragment_index:
            self.save_current_fragment_state()

        self.active_fragment_index = index
        self.load_fragment_state(self.fragments[index])
        self.refresh_tree()
        self.refresh_fragment_buttons()
        self.redraw_canvas()
        self.status.set(f"Открыт фрагмент: {self.fragments[index].name}.")

    def refresh_fragment_buttons(self) -> None:
        self.fragment_count_var.set(f"Фрагменты: {len(self.fragments)}")
        if self.fragment_button_frame is None:
            return

        for child in self.fragment_button_frame.winfo_children():
            child.destroy()
        self.fragment_buttons = []

        for index, fragment in enumerate(self.fragments):
            text = fragment.name
            if index == self.active_fragment_index:
                text = f"> {text}"
            button = ttk.Button(
                self.fragment_button_frame,
                text=text,
                command=lambda selected=index: self.switch_fragment(selected),
                width=17,
            )
            button.grid(row=index, column=0, sticky="ew", padx=2, pady=(0, 4))
            if index == self.active_fragment_index:
                button.state(["disabled"])
            self.fragment_buttons.append(button)

    def on_fragment_frame_configure(self, _event: tk.Event) -> None:
        self.fragment_canvas.configure(scrollregion=self.fragment_canvas.bbox("all"))

    def on_fragment_canvas_configure(self, event: tk.Event) -> None:
        self.fragment_canvas.itemconfigure(self.fragment_window, width=max(1, event.width - 2))

    def next_fragment_name(self) -> str:
        cut_count = sum(1 for fragment in self.fragments if fragment.name != "Исходник")
        return f"Фрагмент {cut_count + 1}"

    def delete_current_fragment(self) -> None:
        if self.active_fragment_index is None or not self.fragments:
            self.status.set("Нет выбранного фрагмента для удаления.")
            return
        if self.active_fragment_index == 0:
            self.status.set("Исходник нельзя удалить: это точка возврата к полной картинке.")
            return

        removed = self.fragments.pop(self.active_fragment_index)
        self.active_fragment_index = None
        self.refresh_fragment_buttons()
        self.switch_fragment(0, save_current=False)
        self.status.set(f"Фрагмент удалён: {removed.name}. Открыт исходник.")

    def source_to_work_point(self, x: float, y: float) -> tuple[float, float]:
        return (
            x * self.source_to_work_scale + self.source_to_work_offset_x,
            y * self.source_to_work_scale + self.source_to_work_offset_y,
        )

    def pdf_source_to_work_point(self, x: float, y: float) -> tuple[float, float]:
        return (
            x * self.pdf_source_to_work_scale + self.pdf_source_to_work_offset_x,
            y * self.pdf_source_to_work_scale + self.pdf_source_to_work_offset_y,
        )

    def work_to_source_point(self, x: float, y: float) -> tuple[float, float]:
        scale = max(self.source_to_work_scale, 0.0001)
        return (
            (x - self.source_to_work_offset_x) / scale,
            (y - self.source_to_work_offset_y) / scale,
        )

    def transform_spot_from_source(self, spot: Hotspot) -> Hotspot:
        x, y = self.source_to_work_point(spot.x, spot.y)
        return Hotspot(
            number=spot.number,
            x=x,
            y=y,
            width=max(spot.width * self.source_to_work_scale, 1),
            height=max(spot.height * self.source_to_work_scale, 1),
            article=spot.article,
            source=spot.source,
        )

    def prepare_ocr_image(self) -> tuple[Path, float, float, float, str]:
        if self.image_path is None:
            raise OcrError("Не открыт файл для OCR")

        if (
            self.pdf_document is None
            or self.original_image is None
            or not self.current_image_is_pdf_page
        ):
            return self.image_path, 1.0, 0.0, 0.0, "изображение"

        fitz = self.get_fitz(show_error=False)
        if fitz is None:
            return self.image_path, 1.0, 0.0, 0.0, "PDF"

        page = self.pdf_document[self.pdf_page_index]
        page_rect = page.rect
        if page_rect.width <= 0 or page_rect.height <= 0:
            return self.image_path, 1.0, 0.0, 0.0, "PDF"

        clip_rect = page_rect
        if self.pdf_source_to_work_scale > 0:
            work_width, work_height = self.original_image.size
            source_left = max(0.0, (0.0 - self.pdf_source_to_work_offset_x) / self.pdf_source_to_work_scale)
            source_top = max(0.0, (0.0 - self.pdf_source_to_work_offset_y) / self.pdf_source_to_work_scale)
            source_right = min(
                page_rect.width * self.pdf_render_scale,
                (work_width - self.pdf_source_to_work_offset_x) / self.pdf_source_to_work_scale,
            )
            source_bottom = min(
                page_rect.height * self.pdf_render_scale,
                (work_height - self.pdf_source_to_work_offset_y) / self.pdf_source_to_work_scale,
            )
            if source_right > source_left + 2 and source_bottom > source_top + 2:
                clip_rect = fitz.Rect(
                    page_rect.x0 + source_left / self.pdf_render_scale,
                    page_rect.y0 + source_top / self.pdf_render_scale,
                    page_rect.x0 + source_right / self.pdf_render_scale,
                    page_rect.y0 + source_bottom / self.pdf_render_scale,
                )

        clip_rect = fitz.Rect(
            max(page_rect.x0, clip_rect.x0),
            max(page_rect.y0, clip_rect.y0),
            min(page_rect.x1, clip_rect.x1),
            min(page_rect.y1, clip_rect.y1),
        )
        if clip_rect.width <= 1 or clip_rect.height <= 1:
            clip_rect = page_rect

        max_side = 5600
        target_scale = min(self.pdf_ocr_render_scale, max_side / max(clip_rect.width, clip_rect.height))
        target_scale = max(self.pdf_render_scale, target_scale)
        if target_scale <= self.pdf_render_scale + 0.05:
            return self.image_path, 1.0, 0.0, 0.0, "PDF"

        matrix = fitz.Matrix(target_scale, target_scale)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False, clip=clip_rect)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        coord_scale = self.pdf_render_scale * self.pdf_source_to_work_scale / target_scale
        clip_source_x = (clip_rect.x0 - page_rect.x0) * self.pdf_render_scale
        clip_source_y = (clip_rect.y0 - page_rect.y0) * self.pdf_render_scale
        coord_offset_x = self.pdf_source_to_work_offset_x + clip_source_x * self.pdf_source_to_work_scale
        coord_offset_y = self.pdf_source_to_work_offset_y + clip_source_y * self.pdf_source_to_work_scale
        image = self.apply_brush_strokes_to_ocr_image(
            image,
            coord_scale,
            coord_offset_x,
            coord_offset_y,
        )
        path = self.write_temp_image(image, f"pdf_ocr_page_{self.pdf_page_index + 1}")
        cropped_note = (
            " crop"
            if abs(clip_rect.width - page_rect.width) > 1 or abs(clip_rect.height - page_rect.height) > 1
            else ""
        )
        return (
            path,
            coord_scale,
            coord_offset_x,
            coord_offset_y,
            f"PDF{cropped_note} x{target_scale:.1f}",
        )

    def open_file(self) -> None:
        file_name = filedialog.askopenfilename(
            title="Выберите изображение или PDF",
            filetypes=(
                ("Images and PDF", "*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.pdf"),
                ("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"),
                ("PDF", "*.pdf"),
                ("All files", "*.*"),
            ),
        )
        if not file_name:
            return

        path = Path(file_name)
        if path.suffix.lower() == ".pdf":
            self.open_pdf(path)
        else:
            self.open_raster_image(path)

    def open_image(self) -> None:
        self.open_file()

    def open_raster_image(self, path: Path) -> None:
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось открыть изображение:\n{exc}")
            return

        self.close_pdf_document()
        self.source_path = path
        self.source_type = "image"
        self.raster_source_image = image.copy()
        self.current_image_is_pdf_page = False
        self.pdf_source_to_work_scale = 1.0
        self.pdf_source_to_work_offset_x = 0.0
        self.pdf_source_to_work_offset_y = 0.0
        self.reset_brush_state()
        self.cutout_regions = []
        self.set_work_image_from_source(image, "image_800")
        self.preview_image = None
        self.preview_photo = None
        self.crop_rect = None
        self.crop_start = None
        self.lasso_points = []
        self.lasso_dragging = False
        self.spots = []
        self.candidate_spots_by_number = {}
        self.current_candidate_options = []
        self.candidate_var.set("")
        self.ocr_run_id += 1
        self.selected_index = None
        self.reset_viewport()
        self.pdf_page_index = 0
        self.pdf_page_count = 0
        self.pdf_page_status.set("PDF не открыт")
        self.refresh_tree()
        self.redraw_canvas()
        self.reset_fragments_from_current("Исходник")
        self.status.set(
            f"Открыто изображение: {path.name}, исходный размер {image.width}x{image.height}, "
            f"рабочий формат {WORK_IMAGE_SIZE}x{WORK_IMAGE_SIZE}."
        )

    def open_pdf(self, path: Path) -> None:
        fitz = self.get_fitz()
        if fitz is None:
            return

        try:
            document = fitz.open(path)
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось открыть PDF:\n{exc}")
            return

        if document.page_count == 0:
            document.close()
            messagebox.showwarning("Пустой PDF", "В PDF нет страниц.")
            return

        self.close_pdf_document()
        self.pdf_document = document
        self.source_path = path
        self.source_type = "pdf"
        self.raster_source_image = None
        self.pdf_page_count = document.page_count
        self.pdf_page_index = 0
        self.load_pdf_page(0)

    def load_pdf_page(self, page_index: int) -> None:
        if self.pdf_document is None:
            return

        page_index = max(0, min(page_index, self.pdf_page_count - 1))
        fitz = self.get_fitz(show_error=False)
        if fitz is None:
            return

        try:
            page = self.pdf_document[page_index]
            matrix = fitz.Matrix(self.pdf_render_scale, self.pdf_render_scale)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        except Exception as exc:
            messagebox.showerror("Ошибка PDF", f"Не удалось отрисовать страницу PDF:\n{exc}")
            return

        self.pdf_page_index = page_index
        self.set_work_image_from_source(image, f"pdf_page_{page_index + 1}_800")
        self.pdf_source_to_work_scale = self.source_to_work_scale
        self.pdf_source_to_work_offset_x = self.source_to_work_offset_x
        self.pdf_source_to_work_offset_y = self.source_to_work_offset_y
        self.current_image_is_pdf_page = True
        self.reset_brush_state()
        self.cutout_regions = []
        self.preview_image = None
        self.preview_photo = None
        self.crop_rect = None
        self.crop_start = None
        self.lasso_points = []
        self.lasso_dragging = False
        self.spots = []
        self.candidate_spots_by_number = {}
        self.current_candidate_options = []
        self.candidate_var.set("")
        self.ocr_run_id += 1
        self.selected_index = None
        self.reset_viewport()
        self.pdf_page_status.set(f"Стр. {page_index + 1} из {self.pdf_page_count}")
        self.refresh_tree()
        self.redraw_canvas()
        self.reset_fragments_from_current("Исходник")
        source_name = self.source_path.name if self.source_path else "PDF"
        self.status.set(
            f"Открыт PDF: {source_name}, страница {page_index + 1} из {self.pdf_page_count}. "
            f"Страница приведена к {WORK_IMAGE_SIZE}x{WORK_IMAGE_SIZE}. "
            "При необходимости обрежьте область, вставьте список и нажмите «Найти по списку»."
        )

    def prev_pdf_page(self) -> None:
        if self.pdf_document is None:
            self.status.set("PDF не открыт.")
            return
        self.load_pdf_page(self.pdf_page_index - 1)

    def next_pdf_page(self) -> None:
        if self.pdf_document is None:
            self.status.set("PDF не открыт.")
            return
        self.load_pdf_page(self.pdf_page_index + 1)

    def parse_and_apply_articles(self) -> None:
        self.articles = parse_articles(self.articles_text.get("1.0", tk.END))
        self.apply_articles_to_spots(update_status=False)
        linked = sum(1 for spot in self.spots if spot.article)
        if self.spots:
            self.status.set(f"Разобрано артикулов: {len(self.articles)}. Привязано: {linked} из {len(self.spots)}.")
        else:
            self.status.set(f"Разобрано артикулов: {len(self.articles)}. Теперь распознайте цифры на PDF или картинке.")

    def bind_articles_to_digits(self) -> None:
        self.articles = parse_articles(self.articles_text.get("1.0", tk.END))
        if not self.articles:
            messagebox.showinfo(
                "Нет артикулов",
                "Вставьте список номеров и артикулов. Например:\n1 ABC-123\n2 456789",
            )
            self.status.set("Список артикулов пустой: привязывать нечего.")
            return

        if self.original_image is None or self.image_path is None:
            messagebox.showinfo("Нет изображения", "Сначала откройте PDF или картинку.")
            return

        fragment_targets = self.fragment_search_indices()
        if fragment_targets:
            self.status.set(
                f"Разобрано номеров: {len(self.articles)}. Запускаю поиск по обрезкам: {len(fragment_targets)}."
            )
            self.search_fragments_by_numbers(fragment_targets)
            return

        removed = len(self.spots)
        self.clear_spots(update_status=False)
        removed_text = f" Старые точки удалены: {removed}." if removed else ""
        self.status.set(
            f"Разобрано артикулов: {len(self.articles)}.{removed_text} "
            "Запускаю полный поиск по списку..."
        )
        self.set_fragment_progress(0.0, "Поиск по изображению")
        self.run_ocr()

    def clear_spots(self, update_status: bool = True) -> None:
        removed = len(self.spots)
        self.spots = []
        self.candidate_spots_by_number = {}
        self.current_candidate_options = []
        self.candidate_var.set("")
        self.ocr_run_id += 1
        self.selected_index = None
        self.dragging_index = None
        self.number_entry.delete(0, tk.END)
        self.article_entry.delete(0, tk.END)
        self.refresh_tree()
        self.redraw_canvas()
        if update_status:
            if removed:
                self.status.set(f"Расстановка очищена: удалено точек {removed}. Можно запустить поиск заново.")
            else:
                self.status.set("Расстановка уже пустая. Можно запустить поиск заново.")

    def rerun_ocr(self) -> None:
        if self.original_image is None or self.image_path is None:
            messagebox.showinfo("Нет изображения", "Сначала откройте PDF или картинку.")
            return

        removed = len(self.spots)
        self.clear_spots(update_status=False)
        if removed:
            self.status.set(f"Удалено старых точек: {removed}. Запускаю повторное распознавание...")
        else:
            self.status.set("Запускаю распознавание заново...")
        self.run_ocr()

    def apply_articles_to_spots(self, update_status: bool = True) -> None:
        self.articles = parse_articles(self.articles_text.get("1.0", tk.END))
        for spot in self.spots:
            spot.article = get_article_for_number(self.articles, spot.number)
        self.refresh_tree()
        self.redraw_canvas()
        if update_status:
            linked = sum(1 for spot in self.spots if spot.article)
            missing = [spot.number for spot in self.spots if not spot.article]
            suffix = f" Не найдены: {', '.join(missing[:12])}." if missing else ""
            self.status.set(f"Привязано артикулов к цифрам: {linked} из {len(self.spots)}.{suffix}")

    def extract_pdf_text_digits(self) -> list[Hotspot]:
        if (
            self.pdf_document is None
            or self.original_image is None
            or not self.current_image_is_pdf_page
        ):
            return []

        page = self.pdf_document[self.pdf_page_index]
        page_rect = page.rect
        if page_rect.width <= 0 or page_rect.height <= 0:
            return []

        known_numbers = set(self.articles.keys())
        spots: list[Hotspot] = []

        try:
            words = page.get_text("words")
        except Exception:
            return []

        for word in words:
            if len(word) < 5:
                continue
            x0, y0, x1, y1, text = word[:5]
            for match in NUMBER_RE.finditer(str(text)):
                number = normalize_number(match.group(0))
                if known_numbers and number not in known_numbers:
                    continue
                source_x = ((x0 + x1) / 2 - page_rect.x0) * self.pdf_render_scale
                source_y = ((y0 + y1) / 2 - page_rect.y0) * self.pdf_render_scale
                x, y = self.pdf_source_to_work_point(source_x, source_y)
                width = max((x1 - x0) * self.pdf_render_scale * self.pdf_source_to_work_scale, 1)
                height = max((y1 - y0) * self.pdf_render_scale * self.pdf_source_to_work_scale, 1)
                spots.append(
                    Hotspot(
                        number=number,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        article=get_article_for_number(self.articles, number),
                        source="PDF text layer",
                    )
                )

        return _dedupe_spots(spots)

    def detect_pdf_table_top_y(self) -> float | None:
        if self.original_image is None or not self.current_image_is_pdf_page:
            return None

        try:
            import numpy as np
        except Exception:
            return None

        gray = np.array(self.original_image.convert("L"))
        height, width = gray.shape[:2]
        if height < 300 or width < 300:
            return None

        bottom_start = int(height * 0.55)
        dark_or_gray = gray < 245
        row_density = dark_or_gray.mean(axis=1)
        rows = [row for row in range(bottom_start, height) if row_density[row] > 0.35]
        if not rows:
            return None

        clusters: list[tuple[int, int]] = []
        start = previous = rows[0]
        for row in rows[1:]:
            if row - previous <= 4:
                previous = row
                continue
            if previous - start >= 8:
                clusters.append((start, previous))
            start = previous = row
        if previous - start >= 8:
            clusters.append((start, previous))

        total_height = sum(bottom - top + 1 for top, bottom in clusters)
        if len(clusters) < 3 or total_height < height * 0.045:
            return None
        table_top = min(top for top, _bottom in clusters)
        return max(height * 0.50, table_top - 10)

    def detect_pdf_header_bottom_y(self) -> float | None:
        if self.original_image is None or not self.current_image_is_pdf_page:
            return None

        try:
            import numpy as np
        except Exception:
            return None

        gray = np.array(self.original_image.convert("L"))
        height, width = gray.shape[:2]
        if height < 300 or width < 300:
            return None

        top_limit = int(height * 0.25)
        dark_or_gray = gray < 245
        row_density = dark_or_gray.mean(axis=1)
        rows = [row for row in range(0, top_limit) if row_density[row] > 0.35]
        if not rows:
            return None

        clusters: list[tuple[int, int]] = []
        start = previous = rows[0]
        for row in rows[1:]:
            if row - previous <= 4:
                previous = row
                continue
            if previous - start >= 10:
                clusters.append((start, previous))
            start = previous = row
        if previous - start >= 10:
            clusters.append((start, previous))

        if not clusters:
            return None
        _top, bottom = max(clusters, key=lambda item: item[1] - item[0])
        return min(height * 0.20, bottom + height * 0.09)

    def clean_ocr_spots(self, spots: list[Hotspot], known_numbers: set[str]) -> list[Hotspot]:
        if self.original_image is None:
            return spots

        image_width = self.original_image.width
        image_height = self.original_image.height
        table_top_y = self.detect_pdf_table_top_y()
        header_bottom_y = self.detect_pdf_header_bottom_y()
        cleaned: list[Hotspot] = []
        for spot in spots:
            normalized = resolve_ocr_number(spot.number, known_numbers)
            if normalized is None:
                continue

            # Titles and model names are usually much larger than callout labels.
            # They can contain valid digits and would otherwise bind to articles.
            if spot.width > image_width * 0.14:
                continue
            if spot.height > image_height * 0.08:
                continue
            if spot.x < 0 or spot.y < 0 or spot.x > image_width or spot.y > image_height:
                continue
            if header_bottom_y is not None and spot.y <= header_bottom_y:
                continue
            if table_top_y is not None and spot.y >= table_top_y:
                continue
            if self.spot_erased_by_brush(spot):
                continue

            spot.number = normalized
            spot.article = get_article_for_number(self.articles, normalized)
            cleaned.append(spot)

        cleaned = self.remove_embedded_single_digits(_dedupe_spots(cleaned))
        cleaned = self.remove_overlapping_conflicts(cleaned)
        return self.collapse_duplicate_numbers(cleaned)

    def remove_embedded_single_digits(self, spots: list[Hotspot]) -> list[Hotspot]:
        result: list[Hotspot] = []
        longer_spots = [spot for spot in spots if len(spot.number) > 1]
        for spot in spots:
            if spot.number:
                embedded = False
                for other in longer_spots:
                    if other is spot or len(spot.number) >= len(other.number):
                        continue
                    if spot.number not in other.number:
                        continue
                    same_line = abs(spot.y - other.y) <= max(spot.height, other.height, 12)
                    inside_x = other.x - other.width / 2 - 2 <= spot.x <= other.x + other.width / 2 + 2
                    if same_line and inside_x:
                        embedded = True
                        break
                if embedded:
                    continue
            result.append(spot)
        return result

    def collapse_duplicate_numbers(self, spots: list[Hotspot]) -> list[Hotspot]:
        by_number: dict[str, list[Hotspot]] = {}
        for spot in spots:
            by_number.setdefault(spot.number, []).append(spot)

        if all(len(items) == 1 for items in by_number.values()):
            return spots

        spacing = self.estimate_label_spacing(spots)
        height = self.estimate_label_height(spots)
        result: list[Hotspot] = []
        for number, items in by_number.items():
            references = [spot for spot in spots if spot.number != number]
            if len(items) == 1:
                self.annotate_candidate_support(items[0], items, number, references, spacing, height)
                result.append(items[0])
                continue

            best = min(
                items,
                key=lambda item: self.candidate_selection_score(
                    number,
                    item,
                    references,
                    spacing,
                    height,
                    items,
                ),
            )
            self.annotate_candidate_support(best, items, number, references, spacing, height)
            result.append(best)

        return sorted(result, key=lambda spot: (int(spot.number) if spot.number.isdigit() else 999999, spot.y, spot.x))

    def remove_overlapping_conflicts(self, spots: list[Hotspot]) -> list[Hotspot]:
        if len(spots) < 2:
            return spots

        kept: list[Hotspot] = []
        for spot in sorted(spots, key=self.spot_quality_score, reverse=True):
            conflict = False
            for existing in kept:
                if existing.number == spot.number:
                    continue
                if self.conflict_family(existing) != self.conflict_family(spot):
                    continue
                distance = ((existing.x - spot.x) ** 2 + (existing.y - spot.y) ** 2) ** 0.5
                conflict_radius = min(
                    6.0,
                    max(2.5, max(existing.width, existing.height, spot.width, spot.height) * 0.35),
                )
                if distance <= conflict_radius:
                    conflict = True
                    break
            if not conflict:
                kept.append(spot)

        return sorted(kept, key=lambda spot: (int(spot.number) if spot.number.isdigit() else 999999, spot.y, spot.x))

    def conflict_family(self, spot: Hotspot) -> str:
        source = spot.source.lower()
        if "opencv digit detector" in source:
            return "opencv"
        if "circled number detector" in source:
            return "circled"
        if "windows ocr" in source:
            return "windows"
        if "tesseract" in source:
            return "tesseract"
        if "pdf text layer" in source:
            return "pdf"
        return source.split(";", 1)[0]

    def spot_quality_score(self, spot: Hotspot) -> float:
        confidence = source_confidence(spot.source)
        if confidence is None:
            confidence = 0.72
        source = spot.source.lower()
        if "pdf text layer" in source:
            confidence += 0.2
        elif "circled number detector" in source:
            confidence += 0.08
        elif "windows ocr" in source:
            confidence += 0.03
        if self.source_has_note(spot, "line"):
            confidence += 0.02
        if self.source_has_note(spot, "sequence"):
            confidence += 0.04
        if "review" in source:
            confidence -= 0.02
        confidence += min(len(spot.number), 4) * 0.02
        return confidence

    def same_number_support_count(self, candidate: Hotspot, candidates: list[Hotspot]) -> int:
        families: set[str] = set()
        radius = max(16.0, candidate.width * 2.5, candidate.height * 2.5)
        for other in candidates:
            distance = ((candidate.x - other.x) ** 2 + (candidate.y - other.y) ** 2) ** 0.5
            if distance <= radius:
                families.add(self.conflict_family(other))
        return len(families)

    def source_selection_penalty(self, candidate: Hotspot) -> float:
        source = candidate.source.lower()
        confidence = source_confidence(candidate.source)
        if "pdf text layer" in source:
            return -4.0
        if "line label ocr" in source:
            return -2.5
        if "dense crop windows ocr" in source:
            return -1.8
        if "windows ocr" in source:
            return -1.5
        if "line label detector" in source:
            return -1.0
        if "circled number detector" in source:
            return -1.0
        if "opencv digit detector" in source:
            return 2.0 if confidence is not None and confidence >= 0.74 else 5.0
        if "800 grid ocr" in source:
            return 7.0
        return 0.0

    def is_weak_pdf_source(self, spot: Hotspot) -> bool:
        source = spot.source.lower()
        return "opencv digit detector" in source or "800 grid ocr" in source

    def source_has_note(self, spot: Hotspot, note: str) -> bool:
        return any(part.strip().lower() == note.lower() for part in spot.source.split(";"))

    def clone_hotspot(self, spot: Hotspot) -> Hotspot:
        return Hotspot(
            number=spot.number,
            x=spot.x,
            y=spot.y,
            width=spot.width,
            height=spot.height,
            article=spot.article,
            source=spot.source,
        )

    def remember_candidate_spots(self, candidates: list[Hotspot]) -> None:
        grouped: dict[str, list[Hotspot]] = {}
        for candidate in candidates:
            number = normalize_number(candidate.number)
            if not number:
                continue
            grouped.setdefault(number, []).append(candidate)

        spacing = self.estimate_label_spacing(candidates)
        height = self.estimate_label_height(candidates)
        result: dict[str, list[Hotspot]] = {}
        for number, items in grouped.items():
            references = [spot for spot in candidates if normalize_number(spot.number) != number]
            ranked = sorted(
                items,
                key=lambda item: self.candidate_selection_score(number, item, references, spacing, height, items),
            )
            unique: list[Hotspot] = []
            for item in ranked:
                if any(((item.x - existing.x) ** 2 + (item.y - existing.y) ** 2) ** 0.5 < 8 for existing in unique):
                    continue
                unique.append(self.clone_hotspot(item))
                if len(unique) >= 12:
                    break
            if unique:
                result[number] = unique
        self.candidate_spots_by_number = result

    def get_leader_lines(self) -> list[LeaderLineSegment]:
        if self.original_image is None or not _opencv_available():
            return []

        key = (str(self.image_path or ""), self.original_image.size)
        if self.leader_line_cache_key == key:
            return self.leader_line_cache

        import cv2
        import numpy as np

        gray = np.array(self.original_image.convert("L"))
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 60, 180)
        raw_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=18,
            minLineLength=14,
            maxLineGap=5,
        )

        segments: list[LeaderLineSegment] = []
        if raw_lines is not None:
            for raw_line in raw_lines[:, 0]:
                x1, y1, x2, y2 = [float(value) for value in raw_line]
                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if length < 14 or length > 360:
                    continue
                segments.append(LeaderLineSegment(x1, y1, x2, y2, length))

        if self.image_path is not None:
            try:
                for box in _find_horizontal_label_boxes(self.image_path):
                    x1 = float(box.line_x)
                    y = float(box.line_y + box.line_height / 2)
                    x2 = float(box.line_x + box.line_width)
                    length = abs(x2 - x1)
                    if length >= 14:
                        segments.append(LeaderLineSegment(x1, y, x2, y, length, "label-line"))
            except Exception:
                pass

        deduped: list[LeaderLineSegment] = []
        for segment in sorted(segments, key=lambda item: item.length, reverse=True):
            if any(self.line_segments_close(segment, existing) for existing in deduped):
                continue
            deduped.append(segment)

        self.leader_line_cache_key = key
        self.leader_line_cache = deduped[:1200]
        return self.leader_line_cache

    def line_segments_close(self, left: LeaderLineSegment, right: LeaderLineSegment) -> bool:
        endpoints_a = ((left.x1, left.y1), (left.x2, left.y2))
        endpoints_b = ((right.x1, right.y1), (right.x2, right.y2))
        direct = sum(
            ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            for (ax, ay), (bx, by) in zip(endpoints_a, endpoints_b)
        )
        reverse = sum(
            ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            for (ax, ay), (bx, by) in zip(endpoints_a, reversed(endpoints_b))
        )
        return min(direct, reverse) <= 10

    def distance_point_to_rect(
        self,
        x: float,
        y: float,
        left: float,
        top: float,
        right: float,
        bottom: float,
    ) -> float:
        dx = max(left - x, 0.0, x - right)
        dy = max(top - y, 0.0, y - bottom)
        return (dx * dx + dy * dy) ** 0.5

    def distance_point_to_segment(self, x: float, y: float, segment: LeaderLineSegment) -> float:
        vx = segment.x2 - segment.x1
        vy = segment.y2 - segment.y1
        length_sq = vx * vx + vy * vy
        if length_sq <= 0.0001:
            return ((x - segment.x1) ** 2 + (y - segment.y1) ** 2) ** 0.5
        t = ((x - segment.x1) * vx + (y - segment.y1) * vy) / length_sq
        t = min(1.0, max(0.0, t))
        px = segment.x1 + t * vx
        py = segment.y1 + t * vy
        return ((x - px) ** 2 + (y - py) ** 2) ** 0.5

    def candidate_line_score(self, candidate: Hotspot) -> float:
        cache_key = (
            round(candidate.x, 1),
            round(candidate.y, 1),
            round(candidate.width, 1),
            round(candidate.height, 1),
            str(self.image_path or ""),
        )
        cached = self.line_score_cache.get(cache_key)
        if cached is not None:
            return cached

        lines = self.get_leader_lines()
        if not lines:
            return 5.0

        half_w = max(candidate.width / 2, 3.0)
        half_h = max(candidate.height / 2, 3.0)
        pad = max(8.0, half_h * 2.0)
        left = candidate.x - half_w - pad
        right = candidate.x + half_w + pad
        top = candidate.y - half_h - pad
        bottom = candidate.y + half_h + pad
        best = 5.0

        for segment in lines:
            endpoint_distance = min(
                self.distance_point_to_rect(segment.x1, segment.y1, left, top, right, bottom),
                self.distance_point_to_rect(segment.x2, segment.y2, left, top, right, bottom),
            )
            if endpoint_distance <= 28:
                best = min(best, endpoint_distance / 28)

            segment_distance = self.distance_point_to_segment(candidate.x, candidate.y, segment)
            if segment_distance <= 16 and endpoint_distance <= 60:
                best = min(best, 0.55 + segment_distance / 32)

            dx = abs(segment.x2 - segment.x1)
            dy = abs(segment.y2 - segment.y1)
            if dx >= max(14.0, dy * 3):
                y = (segment.y1 + segment.y2) / 2
                min_x = min(segment.x1, segment.x2) - 18
                max_x = max(segment.x1, segment.x2) + 18
                if min_x <= candidate.x <= max_x:
                    vertical_gap = min(abs(y - (candidate.y - half_h)), abs(y - (candidate.y + half_h)))
                    if 2 <= vertical_gap <= 64:
                        best = min(best, vertical_gap / 64)

        self.line_score_cache[cache_key] = best
        return best

    def sequence_reference_score(self, spot: Hotspot) -> float:
        score = self.source_selection_penalty(spot) * 0.25
        source = spot.source.lower()
        if self.source_has_note(spot, "line"):
            score -= 1.5
        if "support " in source:
            score -= 1.0
        if "review" in source:
            score += 2.0
        if spot.source.startswith("Inferred"):
            score += 1.5

        line_score = self.candidate_line_score(spot)
        if line_score <= 0.85:
            score -= 1.0
        elif line_score > 2.5:
            score += 1.0
        return score

    def candidate_sequence_score(
        self,
        number: str,
        candidate: Hotspot,
        base_spots: list[Hotspot],
        max_neighbor_span: int = 5,
    ) -> float:
        if not number.isdigit():
            return 5.0

        target = int(number)
        by_number: dict[int, list[Hotspot]] = {}
        for spot in base_spots:
            if not spot.number.isdigit():
                continue
            spot_number = int(spot.number)
            if spot_number == target or abs(spot_number - target) > max_neighbor_span:
                continue
            by_number.setdefault(spot_number, []).append(spot)

        if not by_number:
            return 5.0

        reference_scores: dict[int, float] = {}

        def ref_score(spot: Hotspot) -> float:
            key = id(spot)
            if key not in reference_scores:
                reference_scores[key] = self.sequence_reference_score(spot)
            return reference_scores[key]

        for items in by_number.values():
            items.sort(key=ref_score)
            del items[2:]

        max_step = 90.0 if self.current_image_is_pdf_page else 70.0
        min_step = 4.0
        normalizer = max(candidate.height * 1.6, candidate.width, 8.0)
        scores: list[float] = []

        def add_score(expected_x: float, expected_y: float, step_length: float, reference_penalty: float = 0.0) -> None:
            if not (min_step <= step_length <= max_step):
                return
            distance = ((candidate.x - expected_x) ** 2 + (candidate.y - expected_y) ** 2) ** 0.5
            scores.append(distance / max(step_length, normalizer, 1.0) + reference_penalty)

        left_numbers = sorted(number for number in by_number if number < target)
        right_numbers = sorted((number for number in by_number if number > target), reverse=True)

        for left_number in left_numbers:
            for right_number in right_numbers:
                steps = right_number - left_number
                if steps <= 0 or steps > max_neighbor_span * 2:
                    continue
                ratio = (target - left_number) / steps
                for left_spot in by_number[left_number]:
                    for right_spot in by_number[right_number]:
                        total_distance = ((right_spot.x - left_spot.x) ** 2 + (right_spot.y - left_spot.y) ** 2) ** 0.5
                        step_length = total_distance / steps
                        expected_x = left_spot.x + (right_spot.x - left_spot.x) * ratio
                        expected_y = left_spot.y + (right_spot.y - left_spot.y) * ratio
                        ref_penalty = max(0.0, ref_score(left_spot) + ref_score(right_spot)) * 0.08
                        add_score(expected_x, expected_y, step_length, ref_penalty)

        ordered_numbers = sorted(by_number)
        for first_index, first_number in enumerate(ordered_numbers):
            for second_number in ordered_numbers[first_index + 1 :]:
                if first_number >= target or second_number >= target:
                    continue
                steps = second_number - first_number
                gap = target - second_number
                if steps <= 0 or gap <= 0 or gap > max_neighbor_span:
                    continue
                for first_spot in by_number[first_number]:
                    for second_spot in by_number[second_number]:
                        dx = (second_spot.x - first_spot.x) / steps
                        dy = (second_spot.y - first_spot.y) / steps
                        step_length = (dx * dx + dy * dy) ** 0.5
                        ref_penalty = max(0.0, ref_score(first_spot) + ref_score(second_spot)) * 0.08
                        add_score(second_spot.x + dx * gap, second_spot.y + dy * gap, step_length, ref_penalty)

        for first_index, first_number in enumerate(ordered_numbers):
            for second_number in ordered_numbers[first_index + 1 :]:
                if first_number <= target or second_number <= target:
                    continue
                steps = second_number - first_number
                gap = first_number - target
                if steps <= 0 or gap <= 0 or gap > max_neighbor_span:
                    continue
                for first_spot in by_number[first_number]:
                    for second_spot in by_number[second_number]:
                        dx = (second_spot.x - first_spot.x) / steps
                        dy = (second_spot.y - first_spot.y) / steps
                        step_length = (dx * dx + dy * dy) ** 0.5
                        ref_penalty = max(0.0, ref_score(first_spot) + ref_score(second_spot)) * 0.08
                        add_score(first_spot.x - dx * gap, first_spot.y - dy * gap, step_length, ref_penalty)

        return min(scores) if scores else 5.0

    def append_source_note(self, spot: Hotspot, note: str) -> None:
        if self.source_has_note(spot, note):
            return
        spot.source = f"{spot.source}; {note}"

    def remove_source_note(self, spot: Hotspot, note: str) -> None:
        parts = [part.strip() for part in spot.source.split(";") if part.strip()]
        filtered = [part for part in parts if part.lower() != note.lower()]
        spot.source = "; ".join(filtered) if filtered else spot.source

    def annotate_candidate_support(
        self,
        candidate: Hotspot,
        candidates: list[Hotspot],
        number: str,
        references: list[Hotspot],
        spacing: float,
        height: float,
    ) -> None:
        support = self.same_number_support_count(candidate, candidates)
        line_score = self.candidate_line_score(candidate)
        sequence_score = self.candidate_sequence_score(number, candidate, references)
        if line_score <= 0.85:
            self.append_source_note(candidate, "line")
        if sequence_score <= 1.15:
            self.append_source_note(candidate, "sequence")

        if support >= 2:
            self.remove_source_note(candidate, "review")
            self.append_source_note(candidate, f"support {support}")
            return

        position_score = self.candidate_position_score(number, candidate, references, spacing, height)
        source = candidate.source.lower()
        weak_source = "opencv digit detector" in source or "800 grid ocr" in source
        if (line_score <= 0.85 or sequence_score <= 0.90) and not weak_source:
            self.remove_source_note(candidate, "review")
            return
        if weak_source or (position_score > 3.0 and sequence_score > 2.0) or line_score > 2.5:
            self.append_source_note(candidate, "review")

    def choose_best_ocr_result(
        self,
        results: list[tuple[str, list[Hotspot]]],
        known_numbers: set[str],
    ) -> tuple[str, list[Hotspot]] | None:
        if not results:
            return None

        merged = self.merge_ocr_results(results, known_numbers)
        if merged:
            return "Объединённый OCR", merged

        def score(item: tuple[str, list[Hotspot]]) -> tuple[float, int]:
            name, spots = item
            distinct = {normalize_number(spot.number) for spot in spots}
            duplicates = max(0, len(spots) - len(distinct))
            if known_numbers:
                coverage = len(distinct & known_numbers)
                unknown = len(distinct - known_numbers)
                backend_bonus = 0.3 if name.startswith("Windows OCR") else 0.0
                return coverage * 10 - duplicates * 0.4 - unknown * 0.8 + backend_bonus, len(spots)

            backend_bonus = 4.0 if name.startswith("Windows OCR") else 0.0
            return len(distinct) * 2 - duplicates * 1.5 + backend_bonus, len(spots)

        best_name, best_spots = max(results, key=score)
        if known_numbers:
            other_spots = [spot for name, spots in results if name != best_name for spot in spots]
            self.remember_candidate_spots([*best_spots, *other_spots])
            best_spots = self.supplement_missing_numbers(best_spots, other_spots, known_numbers)
        else:
            self.remember_candidate_spots(best_spots)
        return best_name, best_spots

    def merge_ocr_results(
        self,
        results: list[tuple[str, list[Hotspot]]],
        known_numbers: set[str],
    ) -> list[Hotspot]:
        merged: list[Hotspot] = []
        for _name, spots in results:
            for spot in spots:
                number = resolve_ocr_number(spot.number, known_numbers)
                if number is None:
                    continue
                spot.number = number
                spot.article = get_article_for_number(self.articles, number)
                merged.append(spot)

        if not merged:
            return []

        merged_candidates = self.remove_embedded_single_digits(_dedupe_spots(merged))
        merged_candidates = self.remove_overlapping_conflicts(merged_candidates)
        self.remember_candidate_spots(merged_candidates)
        merged = self.collapse_duplicate_numbers(merged_candidates)
        if known_numbers:
            merged = self.supplement_missing_numbers(merged, merged_candidates, known_numbers)
        return merged

    def supplement_missing_numbers(
        self,
        base_spots: list[Hotspot],
        candidate_spots: list[Hotspot],
        known_numbers: set[str],
    ) -> list[Hotspot]:
        present = {normalize_number(spot.number) for spot in base_spots}
        missing = sorted(
            known_numbers - present,
            key=lambda value: int(value) if value.isdigit() else 999999,
        )
        if not missing or not candidate_spots:
            return base_spots

        spacing = self.estimate_label_spacing(base_spots)
        height = self.estimate_label_height(base_spots)
        supplemented = list(base_spots)

        for number in missing:
            same_number = [spot for spot in candidate_spots if normalize_number(spot.number) == number]
            if not same_number:
                continue

            best_candidate: Hotspot | None = None
            best_score = float("inf")
            for candidate in same_number:
                candidate_score = self.candidate_selection_score(
                    number,
                    candidate,
                    supplemented,
                    spacing,
                    height,
                    same_number,
                )
                if candidate_score < best_score:
                    best_candidate = candidate
                    best_score = candidate_score

            if best_candidate is not None and self.accept_supplement_candidate(best_candidate, best_score):
                best_candidate.article = get_article_for_number(self.articles, number)
                self.annotate_candidate_support(best_candidate, same_number, number, supplemented, spacing, height)
                supplemented.append(best_candidate)

        return _dedupe_spots(supplemented)

    def accept_supplement_candidate(self, candidate: Hotspot, best_score: float) -> bool:
        if best_score <= 2.2:
            return True

        if not self.current_image_is_pdf_page:
            return False

        source = candidate.source.lower()
        if self.candidate_line_score(candidate) <= 0.85 and best_score <= 8.0:
            return True
        if "line label" in source or "label line" in source:
            return True
        if "opencv digit detector" in source and best_score <= 4.0:
            return True
        return False

    def estimate_label_spacing(self, spots: list[Hotspot]) -> float:
        distances: list[float] = []
        numbered = [(int(spot.number), spot) for spot in spots if spot.number.isdigit()]
        for left_number, left_spot in numbered:
            for right_number, right_spot in numbered:
                diff = abs(right_number - left_number)
                if diff == 0 or diff > 3:
                    continue
                if abs(right_spot.y - left_spot.y) > max(left_spot.height, right_spot.height, 12) * 2:
                    continue
                distances.append(abs(right_spot.x - left_spot.x) / diff)
        if not distances:
            return 30.0
        distances.sort()
        return max(8.0, distances[len(distances) // 2])

    def estimate_label_height(self, spots: list[Hotspot]) -> float:
        heights = sorted(spot.height for spot in spots if 4 <= spot.height <= 40)
        if not heights:
            return 12.0
        return max(6.0, heights[len(heights) // 2])

    def candidate_position_score(
        self,
        number: str,
        candidate: Hotspot,
        base_spots: list[Hotspot],
        spacing: float,
        height: float,
    ) -> float:
        if not number.isdigit():
            return 999999.0
        target = int(number)
        scores: list[float] = []
        for reference in base_spots:
            if not reference.number.isdigit():
                continue
            diff = int(reference.number) - target
            if diff == 0 or abs(diff) > 3:
                continue
            expected_x = reference.x - diff * spacing
            expected_y = reference.y
            dx = abs(candidate.x - expected_x) / max(spacing, 1)
            dy = abs(candidate.y - expected_y) / max(height * 2, 1)
            scores.append((dx * dx + dy * dy) ** 0.5)
        if not scores:
            return 999999.0
        return min(scores)

    def candidate_selection_score(
        self,
        number: str,
        candidate: Hotspot,
        base_spots: list[Hotspot],
        spacing: float,
        height: float,
        same_number_candidates: list[Hotspot] | None = None,
    ) -> float:
        score = self.candidate_position_score(number, candidate, base_spots, spacing, height)
        if score > 1000:
            score = 25.0
        confidence = source_confidence(candidate.source)
        if confidence is not None:
            score += max(0.0, 0.90 - confidence) * 6.0
        score += self.source_selection_penalty(candidate)
        line_score = self.candidate_line_score(candidate)
        if line_score <= 0.55:
            score -= 3.0
        elif line_score <= 1.0:
            score -= 1.5
        elif line_score > 2.5 and self.current_image_is_pdf_page:
            score += 1.5
        sequence_score = self.candidate_sequence_score(number, candidate, base_spots)
        if sequence_score <= 0.65:
            score -= 3.5
        elif sequence_score <= 1.25:
            score -= 2.0
        elif sequence_score <= 2.0:
            score -= 0.75
        elif sequence_score > 3.2 and self.current_image_is_pdf_page:
            score += 1.25
        if same_number_candidates:
            support = self.same_number_support_count(candidate, same_number_candidates)
            score -= min(6.0, max(0, support - 1) * 3.0)
            if support <= 1 and self.current_image_is_pdf_page:
                source = candidate.source.lower()
                if "opencv digit detector" in source:
                    score += 4.0
                elif "800 grid ocr" in source:
                    score += 5.0
                if self.is_weak_pdf_source(candidate) and any(
                    not self.is_weak_pdf_source(other) for other in same_number_candidates
                ):
                    score += 20.0
                if line_score > 2.5 and self.is_weak_pdf_source(candidate):
                    score += 5.0
        return score

    def final_position_score(self, number: str, candidate: Hotspot, references: list[Hotspot]) -> float:
        sequence_score = min(self.candidate_sequence_score(number, candidate, references), 5.0)
        line_score = min(self.candidate_line_score(candidate), 5.0)
        score = sequence_score * 2.2 + line_score * 1.2 + self.source_selection_penalty(candidate) * 0.35
        source = candidate.source.lower()
        if self.source_has_note(candidate, "sequence"):
            score -= 0.7
        if self.source_has_note(candidate, "line"):
            score -= 0.8
        if "support " in source:
            score -= 0.6
        if self.is_weak_pdf_source(candidate):
            score += 1.0
        if "review" in source:
            score += 0.5
        if candidate.source.startswith("Inferred"):
            score += 0.8 if line_score <= 1.4 else 3.0
        return score

    def spot_is_reliable_anchor(self, spot: Hotspot) -> bool:
        if not spot.number.isdigit():
            return False
        source = spot.source.lower()
        if spot.source.startswith("Inferred") or "review" in source:
            return False
        if self.is_weak_pdf_source(spot):
            return False
        if self.candidate_line_score(spot) > 1.05:
            return False
        return True

    def sequence_prediction_candidates(
        self,
        number: str,
        references: list[Hotspot],
        width: float,
        height: float,
        max_neighbor_span: int = 8,
    ) -> list[Hotspot]:
        if not number.isdigit():
            return []

        target = int(number)
        by_number: dict[int, list[Hotspot]] = {}
        for spot in references:
            if not spot.number.isdigit():
                continue
            spot_number = int(spot.number)
            if abs(spot_number - target) > max_neighbor_span:
                continue
            by_number.setdefault(spot_number, []).append(spot)

        for items in by_number.values():
            items.sort(key=self.sequence_reference_score)
            del items[2:]

        predictions: list[Hotspot] = []
        max_pair_step = 95.0 if self.current_image_is_pdf_page else 75.0
        max_edge_step = 240.0 if self.current_image_is_pdf_page else 140.0

        def add_prediction(x: float, y: float, source: str) -> None:
            if self.original_image is not None:
                if x < 0 or y < 0 or x > self.original_image.width or y > self.original_image.height:
                    return
            candidate = Hotspot(number=number, x=x, y=y, width=width, height=height, source=source)
            if self.candidate_line_score(candidate) <= 0.95:
                self.append_source_note(candidate, "line")
            self.append_source_note(candidate, "sequence")
            predictions.append(candidate)

        left_numbers = sorted((item for item in by_number if item < target), reverse=True)
        right_numbers = sorted(item for item in by_number if item > target)

        for left_number in left_numbers[:3]:
            for right_number in right_numbers[:3]:
                steps = right_number - left_number
                if steps <= 0 or steps > max_neighbor_span * 2:
                    continue
                ratio = (target - left_number) / steps
                for left_spot in by_number[left_number]:
                    for right_spot in by_number[right_number]:
                        distance = ((right_spot.x - left_spot.x) ** 2 + (right_spot.y - left_spot.y) ** 2) ** 0.5
                        step = distance / steps
                        if 4.0 <= step <= max_pair_step:
                            add_prediction(
                                left_spot.x + (right_spot.x - left_spot.x) * ratio,
                                left_spot.y + (right_spot.y - left_spot.y) * ratio,
                                "Inferred sequence prediction; review",
                            )

        if len(left_numbers) >= 2:
            near_number, far_number = left_numbers[0], left_numbers[1]
            steps = near_number - far_number
            gap = target - near_number
            if steps > 0 and 0 < gap <= max_neighbor_span:
                for near_spot in by_number[near_number]:
                    for far_spot in by_number[far_number]:
                        dx = (near_spot.x - far_spot.x) / steps
                        dy = (near_spot.y - far_spot.y) / steps
                        step = (dx * dx + dy * dy) ** 0.5
                        if 4.0 <= step <= max_edge_step:
                            add_prediction(
                                near_spot.x + dx * gap,
                                near_spot.y + dy * gap,
                                "Inferred sequence edge prediction; review",
                            )

        if len(right_numbers) >= 2:
            near_number, far_number = right_numbers[0], right_numbers[1]
            steps = far_number - near_number
            gap = near_number - target
            if steps > 0 and 0 < gap <= max_neighbor_span:
                for near_spot in by_number[near_number]:
                    for far_spot in by_number[far_number]:
                        dx = (far_spot.x - near_spot.x) / steps
                        dy = (far_spot.y - near_spot.y) / steps
                        step = (dx * dx + dy * dy) ** 0.5
                        if 4.0 <= step <= max_edge_step:
                            add_prediction(
                                near_spot.x - dx * gap,
                                near_spot.y - dy * gap,
                                "Inferred sequence edge prediction; review",
                            )

        return _dedupe_spots(predictions)[:8]

    def refine_final_spot_positions(self, spots: list[Hotspot], known_numbers: set[str]) -> list[Hotspot]:
        if not spots or not known_numbers:
            return spots

        working = list(spots)
        label_height = self.estimate_label_height(working)
        label_widths = sorted(spot.width for spot in working if 4 <= spot.width <= 80)
        label_width = label_widths[len(label_widths) // 2] if label_widths else max(label_height * 1.5, 12.0)

        for _pass in range(2):
            changed = False
            for index, spot in enumerate(list(working)):
                number = normalize_number(spot.number)
                if number not in known_numbers or not number.isdigit():
                    continue

                references = [other for other_index, other in enumerate(working) if other_index != index]
                current_score = self.final_position_score(number, spot, references)
                current_line_score = self.candidate_line_score(spot)
                current_sequence_score = self.candidate_sequence_score(number, spot, references)
                source = spot.source.lower()
                suspicious_marker = (
                    "review" in source
                    or spot.source.startswith("Inferred")
                    or self.is_weak_pdf_source(spot)
                )
                if not suspicious_marker:
                    continue
                if (
                    not spot.source.startswith("Inferred")
                    and current_line_score <= 1.25
                    and current_sequence_score <= 1.8
                    and current_score < 6.0
                ):
                    continue

                options = [self.clone_hotspot(item) for item in self.candidate_spots_by_number.get(number, [])]
                if not options:
                    continue

                best = spot
                best_score = current_score
                for option in options:
                    option.number = number
                    option.article = get_article_for_number(self.articles, number)
                    option_score = self.final_position_score(number, option, references)
                    if option_score < best_score:
                        best = option
                        best_score = option_score

                margin = 0.8 if spot.source.startswith("Inferred") or "review" in source else 1.6
                if best is not spot and current_score - best_score >= margin:
                    replacement = self.clone_hotspot(best)
                    replacement.article = get_article_for_number(self.articles, number)
                    self.append_source_note(replacement, "auto-refined")
                    working[index] = replacement
                    changed = True

            if not changed:
                break

        return self.verify_final_spots(_dedupe_spots(working))

    def verify_final_spots(self, spots: list[Hotspot]) -> list[Hotspot]:
        for index, spot in enumerate(spots):
            if not spot.number.isdigit():
                continue
            references = [other for other_index, other in enumerate(spots) if other_index != index]
            line_score = self.candidate_line_score(spot)
            sequence_score = self.candidate_sequence_score(spot.number, spot, references)
            if line_score <= 0.85:
                self.append_source_note(spot, "line")
            if sequence_score <= 1.15:
                self.append_source_note(spot, "sequence")
            if spot.source.startswith("Inferred"):
                self.append_source_note(spot, "review")
            if "sequence prediction" in spot.source.lower():
                if line_score <= 0.85 and sequence_score <= 1.15:
                    self.remove_source_note(spot, "review")
                elif line_score > 1.6:
                    self.append_source_note(spot, "review")
        return spots

    def park_problem_spots(self, spots: list[Hotspot], known_numbers: set[str]) -> list[Hotspot]:
        if self.original_image is None:
            return spots

        result = list(spots)
        present = {normalize_number(spot.number) for spot in result}
        missing_numbers = sorted(
            known_numbers - present,
            key=lambda value: int(value) if value.isdigit() else 999999,
        )

        label_height = self.estimate_label_height(result)
        label_widths = sorted(spot.width for spot in result if 4 <= spot.width <= 80)
        label_width = label_widths[len(label_widths) // 2] if label_widths else max(label_height * 1.6, 14.0)
        for number in missing_numbers:
            result.append(
                Hotspot(
                    number=number,
                    x=0.0,
                    y=0.0,
                    width=label_width,
                    height=label_height,
                    article=get_article_for_number(self.articles, number),
                    source="Missing placeholder; review",
                )
            )

        problem_indices = [index for index, spot in enumerate(result) if self.spot_is_problem(spot)]
        if not problem_indices:
            return result

        image_width = self.original_image.width
        image_height = self.original_image.height
        cell = 28.0
        start_x = 16.0
        start_y = 16.0
        usable_width = max(cell, min(image_width * 0.42, 260.0))
        columns = max(1, int(usable_width // cell))

        for order, index in enumerate(problem_indices):
            spot = result[index]
            column = order % columns
            row = order // columns
            spot.x = min(image_width - 8.0, start_x + column * cell)
            spot.y = min(image_height - 8.0, start_y + row * cell)
            spot.width = max(spot.width, 18.0)
            spot.height = max(spot.height, 18.0)
            self.append_source_note(spot, "parked")
            self.append_source_note(spot, "review")
        return result

    def scale_ocr_spots(
        self,
        spots: list[Hotspot],
        coord_scale: float,
        offset_x: float,
        offset_y: float,
        source_note: str,
    ) -> list[Hotspot]:
        if abs(coord_scale - 1.0) < 0.001 and abs(offset_x) < 0.001 and abs(offset_y) < 0.001:
            return spots
        for spot in spots:
            spot.x = spot.x * coord_scale + offset_x
            spot.y = spot.y * coord_scale + offset_y
            spot.width *= coord_scale
            spot.height *= coord_scale
            spot.source = f"{spot.source}; {source_note}"
        return spots

    def pdf_text_result_is_useful(self, spots: list[Hotspot], known_numbers: set[str]) -> bool:
        if not spots:
            return False
        distinct = {normalize_number(spot.number) for spot in spots}
        if known_numbers:
            coverage = len(distinct & known_numbers)
            if len(known_numbers) <= 5:
                return coverage == len(known_numbers)
            return coverage >= max(5, int(len(known_numbers) * 0.85))

        numeric = [int(number) for number in distinct if number.isdigit()]
        if not numeric:
            return False
        if len(numeric) == 1 and numeric[0] > 500:
            return False
        return len(distinct) >= 30

    def enough_known_numbers_found(self, spots: list[Hotspot], known_numbers: set[str], ratio: float = 0.90) -> bool:
        if not known_numbers:
            return False
        distinct = {normalize_number(spot.number) for spot in spots}
        return len(distinct & known_numbers) >= max(1, int(len(known_numbers) * ratio))

    def windows_result_has_good_base(
        self,
        results: list[tuple[str, list[Hotspot]]],
        known_numbers: set[str],
    ) -> bool:
        if not known_numbers:
            return False
        minimum = max(3, int(len(known_numbers) * 0.65))
        for name, spots in results:
            if not name.startswith("Windows OCR"):
                continue
            distinct = {normalize_number(spot.number) for spot in spots}
            if len(distinct & known_numbers) >= minimum:
                return True
        return False

    def infer_missing_sequence_spots(self, spots: list[Hotspot], known_numbers: set[str]) -> list[Hotspot]:
        if self.original_image is None or not known_numbers:
            return spots

        present: dict[int, Hotspot] = {}
        for spot in spots:
            if spot.number.isdigit():
                present.setdefault(int(spot.number), spot)

        known_ints = sorted(int(number) for number in known_numbers if number.isdigit())
        missing = [number for number in known_ints if number not in present]
        if not missing:
            return spots

        if self.current_image_is_pdf_page:
            max_step = min(max(min(self.original_image.width, self.original_image.height) * 0.10, 80.0), 150.0)
            edge_max_step = 240.0
            anchor_search_span = 8
        else:
            max_step = min(max(min(self.original_image.width, self.original_image.height) * 0.06, 60.0), 120.0)
            edge_max_step = 140.0
            anchor_search_span = 4
        label_height = self.estimate_label_height(spots)
        label_widths = sorted(spot.width for spot in spots if 4 <= spot.width <= 80)
        label_width = label_widths[len(label_widths) // 2] if label_widths else max(label_height * 1.5, 12.0)
        inferred: list[Hotspot] = []

        index = 0
        while index < len(missing):
            start = missing[index]
            end = start
            while index + 1 < len(missing) and missing[index + 1] == end + 1:
                index += 1
                end = missing[index]

            max_missing_run = 2 if self.current_image_is_pdf_page else 3
            missing_run_length = end - start + 1
            if missing_run_length > max_missing_run:
                index += 1
                continue

            left_number = next(
                (number for number in range(start - 1, max(start - anchor_search_span, -1), -1) if number in present),
                None,
            )
            right_number = next(
                (number for number in range(end + 1, end + anchor_search_span + 1) if number in present),
                None,
            )
            if left_number is None and right_number is None:
                index += 1
                continue

            left_spot = present.get(left_number)
            right_spot = present.get(right_number)
            inferred_from_pair = False
            if left_spot is not None and right_spot is not None:
                steps = right_number - left_number
                distance = ((right_spot.x - left_spot.x) ** 2 + (right_spot.y - left_spot.y) ** 2) ** 0.5
                if (
                    steps > 0
                    and distance / steps <= max_step
                    and self.spot_is_reliable_anchor(left_spot)
                    and self.spot_is_reliable_anchor(right_spot)
                ):
                    for number in range(start, end + 1):
                        ratio = (number - left_number) / steps
                        inferred_spot = Hotspot(
                            number=str(number),
                            x=left_spot.x + (right_spot.x - left_spot.x) * ratio,
                            y=left_spot.y + (right_spot.y - left_spot.y) * ratio,
                            width=label_width,
                            height=label_height,
                            source="Inferred sequence; review",
                        )
                        inferred.append(inferred_spot)
                        present[number] = inferred_spot
                    inferred_from_pair = True
            if not inferred_from_pair and left_spot is not None:
                previous_left_number = next(
                    (
                        number
                        for number in range(left_number - 1, max(left_number - anchor_search_span, -1), -1)
                        if number in present
                    ),
                    None,
                )
                previous_left_spot = present.get(previous_left_number)
                if previous_left_number is not None and previous_left_spot is not None:
                    steps = left_number - previous_left_number
                    dx = (left_spot.x - previous_left_spot.x) / steps
                    dy = (left_spot.y - previous_left_spot.y) / steps
                    step_distance = (dx * dx + dy * dy) ** 0.5
                    if (
                        missing_run_length == 1
                        and steps > 0
                        and step_distance <= max_step
                        and self.spot_is_reliable_anchor(left_spot)
                        and self.spot_is_reliable_anchor(previous_left_spot)
                    ):
                        for number in range(start, end + 1):
                            gap = number - left_number
                            if gap != 1:
                                continue
                            inferred_spot = Hotspot(
                                number=str(number),
                                x=left_spot.x + dx * gap,
                                y=left_spot.y + dy * gap,
                                width=label_width,
                                height=label_height,
                                source="Inferred sequence edge; review",
                            )
                            inferred.append(inferred_spot)
                            present[number] = inferred_spot
                        inferred_from_pair = True
            if not inferred_from_pair and right_spot is not None:
                next_right_number = next(
                    (
                        number
                        for number in range(right_number + 1, right_number + anchor_search_span + 1)
                        if number in present
                    ),
                    None,
                )
                next_right_spot = present.get(next_right_number)
                if next_right_number is not None and next_right_spot is not None:
                    steps = next_right_number - right_number
                    dx = (next_right_spot.x - right_spot.x) / steps
                    dy = (next_right_spot.y - right_spot.y) / steps
                    step_distance = (dx * dx + dy * dy) ** 0.5
                    if (
                        missing_run_length == 1
                        and steps > 0
                        and step_distance <= max_step
                        and self.spot_is_reliable_anchor(right_spot)
                        and self.spot_is_reliable_anchor(next_right_spot)
                    ):
                        for number in range(start, end + 1):
                            gap = right_number - number
                            if gap != 1:
                                continue
                            inferred_spot = Hotspot(
                                number=str(number),
                                x=right_spot.x - dx * gap,
                                y=right_spot.y - dy * gap,
                                width=label_width,
                                height=label_height,
                                source="Inferred sequence edge; review",
                            )
                            inferred.append(inferred_spot)
                            present[number] = inferred_spot

            index += 1

        if not inferred:
            return spots
        return _dedupe_spots([*spots, *inferred])

    def run_ocr(self) -> None:
        if self.image_path is None or self.original_image is None:
            messagebox.showinfo("Нет изображения", "Сначала откройте PDF или картинку.")
            return

        self.articles = parse_articles(self.articles_text.get("1.0", tk.END))
        if not self.articles:
            messagebox.showinfo(
                "Нет списка",
                "Сначала вставьте список номеров и артикулов, затем нажмите «Найти по списку».",
            )
            self.status.set("Поиск не запущен: сначала нужен список номеров и артикулов.")
            return

        known_numbers = set(self.articles.keys())
        self.ocr_run_id += 1
        run_id = self.ocr_run_id

        pdf_spots = self.clean_ocr_spots(self.extract_pdf_text_digits(), known_numbers)
        pdf_count = len({normalize_number(spot.number) for spot in pdf_spots})
        pdf_report = f"PDF text layer: {pdf_count}"
        if self.pdf_text_result_is_useful(pdf_spots, known_numbers):
            self._finish_ocr(
                pdf_spots,
                None,
                "PDF text layer",
                run_id,
                backend_report=(pdf_report,),
            )
            return
        use_pdf_text_as_candidate = False
        if pdf_spots:
            if not known_numbers:
                use_pdf_text_as_candidate = True
            else:
                minimum_text_candidates = max(3, int(len(known_numbers) * 0.15))
                use_pdf_text_as_candidate = pdf_count >= minimum_text_candidates
        initial_results = [("PDF text layer", pdf_spots)] if use_pdf_text_as_candidate else []
        initial_report = [pdf_report]

        available_backends = [item for item in self._ocr_backends() if item.available()]
        if not available_backends:
            messagebox.showwarning(
                "OCR недоступен",
                "Не найден OCR-бэкенд.\n\n"
                "Запустите приложение через run.bat, чтобы установить зависимости.\n"
                "Если нужен Tesseract, установите Tesseract OCR отдельно и добавьте его в PATH.",
            )
            self.ocr_status.set(self._backend_status_text())
            return

        backend_names = ", ".join(backend.name for backend in available_backends)
        try:
            ocr_image_path, coord_scale, coord_offset_x, coord_offset_y, ocr_source = self.prepare_ocr_image()
        except Exception as exc:
            messagebox.showerror("Ошибка OCR", f"Не удалось подготовить изображение для OCR:\n{exc}")
            return

        source_suffix = "" if ocr_source == "изображение" else f" ({ocr_source})"
        self.status.set(f"Распознаю цифры{source_suffix}: {backend_names}. Подождите...")

        def worker() -> None:
            errors: list[str] = []
            backend_report: list[str] = list(initial_report)
            results: list[tuple[str, list[Hotspot]]] = list(initial_results)
            for backend in available_backends:
                if run_id != self.ocr_run_id:
                    return
                use_work_image = isinstance(backend, GridTileOcrBackend)
                backend_image_path = self.image_path if use_work_image and self.image_path is not None else ocr_image_path
                backend_scale = 1.0 if use_work_image else coord_scale
                backend_offset_x = 0.0 if use_work_image else coord_offset_x
                backend_offset_y = 0.0 if use_work_image else coord_offset_y
                backend_source = "800x800 grid" if use_work_image else ocr_source
                result_name = backend.name if backend_source == "изображение" else f"{backend.name} / {backend_source}"
                self.after(
                    0,
                    lambda name=result_name, current_run_id=run_id: (
                        self.status.set(f"{name}: распознаю цифры...")
                        if current_run_id == self.ocr_run_id
                        else None
                    ),
                )
                try:
                    raw_spots = backend.recognize_digits(backend_image_path, known_numbers)
                    raw_spots = self.scale_ocr_spots(
                        raw_spots,
                        backend_scale,
                        backend_offset_x,
                        backend_offset_y,
                        backend_source,
                    )
                    spots = self.clean_ocr_spots(raw_spots, known_numbers)
                except Exception as exc:  # pragma: no cover - background UI path
                    error_message = f"{result_name}: ERROR {exc}"
                    errors.append(error_message)
                    backend_report.append(error_message)
                    continue
                if run_id != self.ocr_run_id:
                    return
                distinct = {normalize_number(spot.number) for spot in spots}
                backend_report.append(f"{result_name}: {len(distinct)}")
                if spots:
                    results.append((result_name, spots))
                    required_ratio = 0.90
                    if self.current_image_is_pdf_page and len(known_numbers) >= 30:
                        required_ratio = 0.90 if result_name.startswith("Line label detector") else 0.97
                    if known_numbers and self.enough_known_numbers_found(spots, known_numbers, 1.0):
                        self.after(
                            0,
                            lambda result=spots,
                            name=result_name,
                            current_run_id=run_id,
                            report=tuple(backend_report),
                            backend_errors=tuple(errors): self._finish_ocr(
                                result,
                                None,
                                name,
                                current_run_id,
                                backend_report=report,
                                backend_errors=backend_errors,
                            ),
                        )
                        return

                    combined = self.choose_best_ocr_result(results, known_numbers)
                    combined_ratio = 0.90
                    if self.current_image_is_pdf_page and len(known_numbers) >= 30:
                        combined_ratio = 0.90 if result_name.startswith("Line label detector") else 0.97
                    if (
                        combined is not None
                        and self.windows_result_has_good_base(results, known_numbers)
                        and self.enough_known_numbers_found(combined[1], known_numbers, combined_ratio)
                    ):
                        combined_name, combined_spots = combined
                        self.after(
                            0,
                            lambda result=combined_spots,
                            name=combined_name,
                            current_run_id=run_id,
                            report=tuple(backend_report),
                            backend_errors=tuple(errors): self._finish_ocr(
                                result,
                                None,
                                name,
                                current_run_id,
                                backend_report=report,
                                backend_errors=backend_errors,
                            ),
                        )
                        return

            best = self.choose_best_ocr_result(results, known_numbers)
            if best is not None:
                best_name, best_spots = best
                self.after(
                    0,
                    lambda result=best_spots,
                    name=best_name,
                    current_run_id=run_id,
                    report=tuple(backend_report),
                    backend_errors=tuple(errors): self._finish_ocr(
                        result,
                        None,
                        name,
                        current_run_id,
                        backend_report=report,
                        backend_errors=backend_errors,
                    ),
                )
                return

            error = OcrError("\n".join(errors)) if errors and len(errors) == len(available_backends) else None
            self.after(
                0,
                lambda current_run_id=run_id,
                report=tuple(backend_report),
                backend_errors=tuple(errors): self._finish_ocr(
                    [],
                    error,
                    "OCR",
                    current_run_id,
                    backend_report=report,
                    backend_errors=backend_errors,
                ),
            )

        threading.Thread(target=worker, daemon=True).start()

    def finalize_ocr_spots_for_current(self, spots: list[Hotspot], known_numbers: set[str]) -> list[Hotspot]:
        spots = self.infer_missing_sequence_spots(spots, known_numbers)
        if not self.candidate_spots_by_number:
            self.remember_candidate_spots(spots)
        spots = self.refine_final_spot_positions(spots, known_numbers)
        for spot in spots:
            spot.article = get_article_for_number(self.articles, spot.number)
        return self.park_problem_spots(spots, known_numbers)

    def recognize_current_image_sync(
        self,
        known_numbers: set[str],
        update_ui: bool = True,
    ) -> tuple[str, list[Hotspot], tuple[str, ...], tuple[str, ...]]:
        if self.image_path is None or self.original_image is None:
            raise OcrError("Нет изображения для OCR")

        self.candidate_spots_by_number = {}
        self.current_candidate_options = []
        if update_ui:
            self.candidate_var.set("")
        self.leader_line_cache_key = None
        self.leader_line_cache = []
        self.line_score_cache = {}

        pdf_spots = self.clean_ocr_spots(self.extract_pdf_text_digits(), known_numbers)
        pdf_count = len({normalize_number(spot.number) for spot in pdf_spots})
        pdf_report = f"PDF text layer: {pdf_count}"
        if self.pdf_text_result_is_useful(pdf_spots, known_numbers):
            return (
                "PDF text layer",
                self.finalize_ocr_spots_for_current(pdf_spots, known_numbers),
                (pdf_report,),
                (),
            )

        use_pdf_text_as_candidate = False
        if pdf_spots:
            if not known_numbers:
                use_pdf_text_as_candidate = True
            else:
                minimum_text_candidates = max(3, int(len(known_numbers) * 0.15))
                use_pdf_text_as_candidate = pdf_count >= minimum_text_candidates
        results: list[tuple[str, list[Hotspot]]] = [("PDF text layer", pdf_spots)] if use_pdf_text_as_candidate else []
        backend_report: list[str] = [pdf_report]
        errors: list[str] = []

        available_backends = [item for item in self._ocr_backends() if item.available()]
        if not available_backends:
            raise OcrError("OCR backend не найден")

        ocr_image_path, coord_scale, coord_offset_x, coord_offset_y, ocr_source = self.prepare_ocr_image()
        for backend in available_backends:
            use_work_image = isinstance(backend, GridTileOcrBackend)
            backend_image_path = self.image_path if use_work_image and self.image_path is not None else ocr_image_path
            backend_scale = 1.0 if use_work_image else coord_scale
            backend_offset_x = 0.0 if use_work_image else coord_offset_x
            backend_offset_y = 0.0 if use_work_image else coord_offset_y
            backend_source = "800x800 grid" if use_work_image else ocr_source
            result_name = backend.name if use_work_image else f"{backend.name} / {backend_source}"
            try:
                raw_spots = backend.recognize_digits(backend_image_path, known_numbers)
                raw_spots = self.scale_ocr_spots(
                    raw_spots,
                    backend_scale,
                    backend_offset_x,
                    backend_offset_y,
                    backend_source,
                )
                spots = self.clean_ocr_spots(raw_spots, known_numbers)
            except Exception as exc:
                error_message = f"{result_name}: ERROR {exc}"
                errors.append(error_message)
                backend_report.append(error_message)
                continue

            distinct = {normalize_number(spot.number) for spot in spots}
            backend_report.append(f"{result_name}: {len(distinct)}")
            if spots:
                results.append((result_name, spots))
                if known_numbers and self.enough_known_numbers_found(spots, known_numbers, 1.0):
                    return (
                        result_name,
                        self.finalize_ocr_spots_for_current(spots, known_numbers),
                        tuple(backend_report),
                        tuple(errors),
                    )

                combined = self.choose_best_ocr_result(results, known_numbers)
                if (
                    combined is not None
                    and self.windows_result_has_good_base(results, known_numbers)
                    and self.enough_known_numbers_found(combined[1], known_numbers, 0.90)
                ):
                    combined_name, combined_spots = combined
                    return (
                        combined_name,
                        self.finalize_ocr_spots_for_current(combined_spots, known_numbers),
                        tuple(backend_report),
                        tuple(errors),
                    )

        best = self.choose_best_ocr_result(results, known_numbers)
        if best is None:
            return ("OCR", [], tuple(backend_report), tuple(errors))

        best_name, best_spots = best
        return (
            best_name,
            self.finalize_ocr_spots_for_current(best_spots, known_numbers),
            tuple(backend_report),
            tuple(errors),
        )

    def fragment_search_indices(self) -> list[int]:
        if len(self.fragments) > 1:
            return list(range(1, len(self.fragments)))
        return []

    def safe_fragment_stem(self, fragment: ImageFragment, index: int) -> str:
        raw = fragment.name.strip() or f"fragment_{index + 1}"
        chars = [char if char.isalnum() else "_" for char in raw]
        stem = re.sub(r"_+", "_", "".join(chars)).strip("_")
        return stem or f"fragment_{index + 1}"

    def fragment_results_dir(self) -> Path:
        output_dir = application_dir() / "fragment_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def write_fragment_result_json(
        self,
        index: int,
        fragment: ImageFragment,
        backend_name: str,
        backend_report: tuple[str, ...],
        backend_errors: tuple[str, ...],
        known_numbers: set[str],
    ) -> Path:
        output_dir = self.fragment_results_dir()
        stem = self.safe_fragment_stem(fragment, index)
        image_path = output_dir / f"{stem}.png"
        json_path = output_dir / f"{stem}.json"
        fragment.image.save(image_path)

        spots = sorted(
            fragment.spots,
            key=lambda spot: int(spot.number) if spot.number.isdigit() else 999999,
        )
        found_numbers = {normalize_number(spot.number) for spot in spots}
        missing_numbers = sorted(
            known_numbers - found_numbers,
            key=lambda value: int(value) if value.isdigit() else 999999,
        )
        payload = {
            "build": APP_BUILD,
            "fragment": fragment.name,
            "fragment_index": index,
            "image_file": image_path.name,
            "image_width": fragment.image.width,
            "image_height": fragment.image.height,
            "source": str(self.source_path) if self.source_path else "",
            "source_type": self.source_type,
            "pdf_page": self.pdf_page_index + 1 if self.source_type == "pdf" else None,
            "selected_result": backend_name,
            "requested_numbers": sorted(
                known_numbers,
                key=lambda value: int(value) if value.isdigit() else 999999,
            ),
            "found_numbers": sorted(
                found_numbers,
                key=lambda value: int(value) if value.isdigit() else 999999,
            ),
            "missing_numbers": missing_numbers,
            "backend_report": list(backend_report),
            "backend_errors": list(backend_errors),
            "spots": [
                {
                    "number": spot.number,
                    "article": spot.article,
                    "x_px": round(spot.x, 2),
                    "y_px": round(spot.y, 2),
                    "width_px": round(spot.width, 2),
                    "height_px": round(spot.height, 2),
                    "x_percent": round(spot.x / max(fragment.image.width, 1) * 100, 4),
                    "y_percent": round(spot.y / max(fragment.image.height, 1) * 100, 4),
                    "source": spot.source,
                }
                for spot in spots
            ],
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return json_path

    def search_fragments_by_numbers(self, target_indices: list[int]) -> None:
        if self.original_image is None or self.image_path is None:
            messagebox.showinfo("Нет изображения", "Сначала откройте PDF или картинку.")
            return
        if not target_indices:
            self.run_ocr()
            return
        if self.fragment_search_running:
            self.status.set("Поиск по обрезкам уже идёт. Дождитесь завершения.")
            return

        known_numbers = set(self.articles.keys())
        self.save_current_fragment_state()
        original_index = self.active_fragment_index if self.active_fragment_index is not None else 0
        self.fragment_search_running = True
        self.set_fragment_progress(0.0, f"0/{len(target_indices)}")
        self.status.set(
            f"Поиск по обрезкам запущен в фоне: фрагментов {len(target_indices)}, номеров {len(known_numbers)}."
        )
        threading.Thread(
            target=self._search_fragments_by_numbers_worker,
            args=(list(target_indices), known_numbers, original_index),
            daemon=True,
        ).start()

    def set_status_from_worker(self, text: str) -> None:
        self.enqueue_ui(lambda: self.status.set(text))

    def set_fragment_progress(self, value: float, text: str) -> None:
        self.fragment_progress_var.set(max(0.0, min(100.0, value)))
        self.fragment_progress_text.set(text)

    def set_fragment_progress_from_worker(self, value: float, text: str) -> None:
        self.enqueue_ui(lambda: self.set_fragment_progress(value, text))

    def _search_fragments_by_numbers_worker(
        self,
        target_indices: list[int],
        known_numbers: set[str],
        original_index: int,
    ) -> None:
        found_indices: list[int] = []
        json_paths: list[Path] = []
        errors: list[str] = []

        try:
            for position, index in enumerate(target_indices, start=1):
                if not (0 <= index < len(self.fragments)):
                    continue
                fragment_name = self.fragments[index].name
                progress_before = (position - 1) / max(len(target_indices), 1) * 100
                self.set_fragment_progress_from_worker(
                    progress_before,
                    f"{position}/{len(target_indices)}: {fragment_name}",
                )
                self.set_status_from_worker(
                    f"Поиск по обрезкам: {position} из {len(target_indices)} - {fragment_name}"
                )

                self.active_fragment_index = index
                self.load_fragment_state(self.fragments[index], update_ui=False)
                self.spots = []
                self.candidate_spots_by_number = {}
                self.current_candidate_options = []
                try:
                    backend_name, spots, backend_report, backend_errors = self.recognize_current_image_sync(
                        known_numbers,
                        update_ui=False,
                    )
                except Exception as exc:
                    errors.append(f"{fragment_name}: {exc}")
                    self.spots = []
                    self.save_current_fragment_state()
                    progress_after = position / max(len(target_indices), 1) * 100
                    self.set_fragment_progress_from_worker(
                        progress_after,
                        f"{position}/{len(target_indices)} ошибка",
                    )
                    continue

                self.spots = spots
                self.save_current_fragment_state()
                if spots:
                    found_indices.append(index)
                    json_paths.append(
                        self.write_fragment_result_json(
                            index,
                            self.fragments[index],
                            backend_name,
                            backend_report,
                            backend_errors,
                            known_numbers,
                        )
                    )
                progress_after = position / max(len(target_indices), 1) * 100
                self.set_fragment_progress_from_worker(
                    progress_after,
                    f"{position}/{len(target_indices)} готово",
                )
        except Exception as exc:
            errors.append(f"batch: {exc}")

        self.enqueue_ui(
            lambda: self.finish_fragment_search(
                target_indices,
                found_indices,
                json_paths,
                errors,
                original_index,
            ),
        )

    def finish_fragment_search(
        self,
        target_indices: list[int],
        found_indices: list[int],
        json_paths: list[Path],
        errors: list[str],
        original_index: int,
    ) -> None:
        self.fragment_search_running = False
        self.set_fragment_progress(100.0, f"Готово: {len(found_indices)}/{len(target_indices)}")
        if found_indices:
            self.switch_fragment(found_indices[0], save_current=False)
        elif 0 <= original_index < len(self.fragments):
            self.switch_fragment(original_index, save_current=False)
        self.refresh_fragment_buttons()

        found_spots = sum(len(self.fragments[index].spots) for index in found_indices)
        suffix = f" JSON: {self.fragment_results_dir()}" if json_paths else ""
        error_suffix = f" Ошибок: {len(errors)}." if errors else ""
        self.status.set(
            f"Поиск по обрезкам завершён: найдено фрагментов {len(found_indices)} из {len(target_indices)}, "
            f"точек {found_spots}, JSON файлов {len(json_paths)}.{suffix}{error_suffix}"
        )

    def _finish_ocr(
        self,
        spots: list[Hotspot],
        error: Exception | None,
        backend_name: str,
        run_id: int | None = None,
        backend_report: tuple[str, ...] | list[str] = (),
        backend_errors: tuple[str, ...] | list[str] = (),
    ) -> None:
        if run_id is not None and run_id != self.ocr_run_id:
            return

        if error is not None:
            self.write_ocr_report(backend_name, [], backend_report, backend_errors, [], error)
            messagebox.showerror("Ошибка OCR", str(error))
            self.set_fragment_progress(0.0, "Ошибка OCR")
            self.status.set(f"{backend_name}: распознавание не выполнено.")
            return

        known_numbers = set(self.articles.keys())
        spots = self.infer_missing_sequence_spots(spots, known_numbers)
        if not self.candidate_spots_by_number:
            self.remember_candidate_spots(spots)
        spots = self.refine_final_spot_positions(spots, known_numbers)
        for spot in spots:
            spot.article = get_article_for_number(self.articles, spot.number)
        spots = self.park_problem_spots(spots, known_numbers)
        self.spots = spots
        self.selected_index = None
        self.refresh_tree()
        self.redraw_canvas()
        self.set_fragment_progress(100.0, f"Готово: {len(spots)} точек")
        linked = sum(1 for spot in self.spots if spot.article)
        inferred = sum(1 for spot in self.spots if spot.source.startswith("Inferred"))
        review = sum(1 for spot in self.spots if "review" in spot.source.lower())
        parked = sum(1 for spot in self.spots if self.source_has_note(spot, "parked"))
        line_supported = sum(1 for spot in self.spots if self.source_has_note(spot, "line"))
        inferred_suffix = f", вычислено по соседям {inferred}" if inferred else ""
        line_suffix = f", с линией {line_supported}" if line_supported else ""
        if not self.articles:
            self.write_ocr_report(backend_name, self.spots, backend_report, backend_errors, [])
            self.status.set(
                f"{backend_name}: найдено цифр {len(spots)}{inferred_suffix}. "
                "Артикулы не привязаны: вставьте список и нажмите «Найти по списку»."
            )
            return

        missing_articles = [spot.number for spot in self.spots if not spot.article]
        suffix = f" Не найдены артикулы для: {', '.join(missing_articles[:12])}." if missing_articles else ""
        found_numbers = {normalize_number(spot.number) for spot in self.spots}
        missing_numbers = sorted(
            set(self.articles.keys()) - found_numbers,
            key=lambda value: int(value) if value.isdigit() else 999999,
        )
        if missing_numbers:
            suffix += f" Не найдены номера: {', '.join(missing_numbers[:20])}."
        expected = len(self.articles)
        report_path = self.write_ocr_report(
            backend_name,
            self.spots,
            backend_report,
            backend_errors,
            missing_numbers,
        )
        weak_result = expected >= 30 and linked / expected < 0.5
        if weak_result:
            suffix += " Слабый результат: закройте старые окна и смотрите last_ocr_report.txt."
        if review:
            suffix += f" На проверку выделено {review}."
        if parked:
            suffix += f" Parked in top-left: {parked}."
        if report_path is not None:
            suffix += f" Отчёт: {report_path.name}."
        self.status.set(
            f"{backend_name}: найдено цифр {len(spots)}, с артикулами {linked} из {expected}"
            f"{inferred_suffix}{line_suffix}.{suffix}"
        )
        if weak_result:
            messagebox.showwarning(
                "Слабый результат OCR",
                f"Привязано {linked} из {expected}.\n\n"
                f"Если в заголовке нет build {APP_BUILD}, закройте старое окно и запустите run.bat заново.\n"
                "Если build верный, откройте last_ocr_report.txt: там видно, какой OCR-проход упал или дал мало цифр.",
            )

    def write_ocr_report(
        self,
        backend_name: str,
        spots: list[Hotspot],
        backend_report: tuple[str, ...] | list[str],
        backend_errors: tuple[str, ...] | list[str],
        missing_numbers: list[str],
        error: Exception | None = None,
    ) -> Path | None:
        found_numbers = sorted(
            {normalize_number(spot.number) for spot in spots},
            key=lambda value: int(value) if value.isdigit() else 999999,
        )
        inferred_numbers = sorted(
            {spot.number for spot in spots if spot.source.startswith("Inferred")},
            key=lambda value: int(value) if value.isdigit() else 999999,
        )
        review_numbers = sorted(
            {spot.number for spot in spots if "review" in spot.source.lower()},
            key=lambda value: int(value) if value.isdigit() else 999999,
        )
        line_numbers = sorted(
            {spot.number for spot in spots if self.source_has_note(spot, "line")},
            key=lambda value: int(value) if value.isdigit() else 999999,
        )
        sequence_numbers = sorted(
            {spot.number for spot in spots if self.source_has_note(spot, "sequence")},
            key=lambda value: int(value) if value.isdigit() else 999999,
        )
        parked_numbers = sorted(
            {spot.number for spot in spots if self.source_has_note(spot, "parked")},
            key=lambda value: int(value) if value.isdigit() else 999999,
        )
        linked = sum(1 for spot in spots if spot.article)
        inferred = sum(1 for spot in spots if spot.source.startswith("Inferred"))
        review = sum(1 for spot in spots if "review" in spot.source.lower())
        line_supported = sum(1 for spot in spots if self.source_has_note(spot, "line"))
        sequence_supported = sum(1 for spot in spots if self.source_has_note(spot, "sequence"))
        parked = sum(1 for spot in spots if self.source_has_note(spot, "parked"))
        candidate_total = sum(len(items) for items in self.candidate_spots_by_number.values())
        source = str(self.source_path or self.image_path or "")
        page = self.pdf_page_index + 1 if self.source_type == "pdf" else "-"
        report_path = application_dir() / "last_ocr_report.txt"
        lines = [
            f"Build: {APP_BUILD}",
            f"Source: {source}",
            f"Source type: {self.source_type}",
            f"PDF page: {page}",
            f"Selected result: {backend_name}",
            f"Brush strokes: {len(self.brush_strokes)}",
            f"Candidate numbers: {len(self.candidate_spots_by_number)}",
            f"Candidate spots: {candidate_total}",
            f"Expected article count: {len(self.articles)}",
            f"Unique numbers found: {len(found_numbers)}",
            f"Linked spots: {linked}",
            f"Total spots: {len(spots)}",
            f"Inferred spots: {inferred}",
            f"Review spots: {review}",
            f"Parked problem spots: {parked}",
            f"Line-supported spots: {line_supported}",
            f"Sequence-supported spots: {sequence_supported}",
            f"Inferred numbers: {', '.join(inferred_numbers) if inferred_numbers else '-'}",
            f"Review numbers: {', '.join(review_numbers) if review_numbers else '-'}",
            f"Parked numbers: {', '.join(parked_numbers) if parked_numbers else '-'}",
            f"Line-supported numbers: {', '.join(line_numbers) if line_numbers else '-'}",
            f"Sequence-supported numbers: {', '.join(sequence_numbers) if sequence_numbers else '-'}",
            f"Missing numbers: {', '.join(missing_numbers) if missing_numbers else '-'}",
            "",
            "Backend results:",
        ]
        if backend_report:
            lines.extend(f"- {line}" for line in backend_report)
        else:
            lines.append("- no backend report")
        if backend_errors:
            lines.extend(["", "Backend errors:"])
            lines.extend(f"- {line}" for line in backend_errors)
        if error is not None:
            lines.extend(["", "Final error:", str(error)])
        lines.extend(["", "Found numbers:", ", ".join(found_numbers) if found_numbers else "-"])
        if spots and self.original_image is not None:
            lines.extend(["", "Spot details:"])
            for spot in sorted(spots, key=lambda item: int(item.number) if item.number.isdigit() else 999999):
                x_percent, y_percent = spot.center_percent(self.original_image.width, self.original_image.height)
                lines.append(f"- {spot.number}: x={x_percent:.2f} y={y_percent:.2f} source={spot.source}")
        try:
            report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            return None
        return report_path

    def refresh_tree(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        if self.original_image is None:
            return

        for index, spot in enumerate(self.spots):
            if not self.spot_visible(spot):
                continue
            x_percent, y_percent = spot.center_percent(self.original_image.width, self.original_image.height)
            article_label = spot.article if len(spot.article) <= 42 else spot.article[:39] + "..."
            self.tree.insert(
                "",
                tk.END,
                iid=str(index),
                values=(spot.number, f"{x_percent:.2f}", f"{y_percent:.2f}", article_label),
            )

        if self.selected_index is not None and 0 <= self.selected_index < len(self.spots):
            iid = str(self.selected_index)
            if self.tree.exists(iid):
                self.tree.selection_set(iid)

    def spot_is_review(self, spot: Hotspot) -> bool:
        return "review" in spot.source.lower()

    def spot_is_inferred(self, spot: Hotspot) -> bool:
        return spot.source.startswith("Inferred")

    def spot_is_confirmed_inferred(self, spot: Hotspot) -> bool:
        return (
            self.spot_is_inferred(spot)
            and not self.spot_is_review(spot)
            and self.source_has_note(spot, "line")
            and self.source_has_note(spot, "sequence")
            and bool(spot.article)
        )

    def spot_is_problem(self, spot: Hotspot) -> bool:
        if not spot.article or spot.source.startswith("Missing placeholder"):
            return True
        if self.spot_is_inferred(spot):
            return not self.spot_is_confirmed_inferred(spot)
        if self.spot_is_review(spot):
            source = spot.source.lower()
            if self.source_has_note(spot, "line") and self.source_has_note(spot, "sequence"):
                return False
            if "support " in source and self.source_has_note(spot, "line"):
                return False
            return True
        return False

    def spot_visible(self, spot: Hotspot) -> bool:
        if self.problem_mode.get():
            return self.spot_is_problem(spot)

        current_filter = self.spot_filter.get()
        if current_filter == "Проблемные":
            return self.spot_is_problem(spot)
        if current_filter == "Розовые":
            return self.spot_is_review(spot)
        if current_filter == "Вычисленные":
            return self.spot_is_inferred(spot)
        if current_filter == "Жёлтые":
            return bool(spot.article) and not self.spot_is_review(spot) and (
                not self.spot_is_inferred(spot) or self.spot_is_confirmed_inferred(spot)
            )
        if current_filter == "Без артикула":
            return not spot.article
        if current_filter == "С линией":
            return self.source_has_note(spot, "line")
        return True

    def visible_spot_indices(self) -> list[int]:
        return [index for index, spot in enumerate(self.spots) if self.spot_visible(spot)]

    def problem_spot_indices(self) -> list[int]:
        return [index for index, spot in enumerate(self.spots) if self.spot_is_problem(spot)]

    def on_filter_changed(self, _event: tk.Event | None = None) -> None:
        if self.spot_filter.get() != "Проблемные":
            self.problem_mode.set(False)
        self.refresh_tree()
        self.redraw_canvas()

    def toggle_problem_mode(self) -> None:
        if self.problem_mode.get():
            self.spot_filter.set("Проблемные")
            self.status.set("Режим проверки: показаны только проблемные точки.")
        self.refresh_tree()
        self.redraw_canvas()

    def center_view_on_point(self, x: float, y: float) -> None:
        if self.original_image is None:
            return
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        image_width, image_height = self.original_image.size
        fit_scale = self.fit_preview_scale(canvas_width, canvas_height)
        if self.viewport_zoom < 4.0:
            self.viewport_zoom = min(4.0, max(1.0, 8.0 / max(fit_scale, 0.0001)))
        scale = fit_scale * self.viewport_zoom
        preview_width = max(1, int(image_width * scale))
        preview_height = max(1, int(image_height * scale))
        base_x = (canvas_width - preview_width) / 2
        base_y = (canvas_height - preview_height) / 2
        self.viewport_pan_x = canvas_width / 2 - x * scale - base_x
        self.viewport_pan_y = canvas_height / 2 - y * scale - base_y
        self.clamp_viewport_pan(canvas_width, canvas_height, preview_width, preview_height)

    def select_next_from_indices(self, indices: list[int], direction: int) -> None:
        if not indices:
            self.status.set("Нет точек для выбранного режима.")
            return
        if self.selected_index in indices:
            position = indices.index(self.selected_index)
            target = indices[(position + direction) % len(indices)]
        elif direction >= 0:
            target = indices[0]
        else:
            target = indices[-1]
        self.select_spot(target, center=True)

    def next_problem_spot(self) -> None:
        self.problem_mode.set(True)
        self.spot_filter.set("Проблемные")
        self.refresh_tree()
        self.select_next_from_indices(self.problem_spot_indices(), 1)

    def prev_problem_spot(self) -> None:
        self.problem_mode.set(True)
        self.spot_filter.set("Проблемные")
        self.refresh_tree()
        self.select_next_from_indices(self.problem_spot_indices(), -1)

    def fit_preview_scale(self, canvas_width: int, canvas_height: int) -> float:
        if self.original_image is None:
            return 1.0
        image_width, image_height = self.original_image.size
        return min(canvas_width / image_width, canvas_height / image_height, 1.0)

    def clamp_viewport_pan(self, canvas_width: int, canvas_height: int, preview_width: int, preview_height: int) -> None:
        base_x = (canvas_width - preview_width) / 2
        base_y = (canvas_height - preview_height) / 2

        if preview_width <= canvas_width:
            min_pan_x = -base_x
            max_pan_x = canvas_width - preview_width - base_x
        else:
            min_pan_x = canvas_width - preview_width - base_x
            max_pan_x = -base_x

        if preview_height <= canvas_height:
            min_pan_y = -base_y
            max_pan_y = canvas_height - preview_height - base_y
        else:
            min_pan_y = canvas_height - preview_height - base_y
            max_pan_y = -base_y

        self.viewport_pan_x = min(max(self.viewport_pan_x, min_pan_x), max_pan_x)
        self.viewport_pan_y = min(max(self.viewport_pan_y, min_pan_y), max_pan_y)

    def redraw_canvas(self, redraw_minimap: bool = True) -> None:
        self.canvas.delete("all")
        if self.original_image is None:
            self.canvas.create_text(
                self.canvas.winfo_width() / 2,
                self.canvas.winfo_height() / 2,
                text="Откройте изображение",
                fill="#667085",
                font=("Segoe UI", 15),
            )
            return

        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        image_width, image_height = self.original_image.size
        scale = self.fit_preview_scale(canvas_width, canvas_height) * self.viewport_zoom
        preview_size = (max(1, int(image_width * scale)), max(1, int(image_height * scale)))
        self.clamp_viewport_pan(canvas_width, canvas_height, *preview_size)

        self.preview_scale = scale
        self.preview_offset_x = (canvas_width - preview_size[0]) / 2 + self.viewport_pan_x
        self.preview_offset_y = (canvas_height - preview_size[1]) / 2 + self.viewport_pan_y

        source_left = max(0, int((0 - self.preview_offset_x) / max(scale, 0.0001)) - 2)
        source_top = max(0, int((0 - self.preview_offset_y) / max(scale, 0.0001)) - 2)
        source_right = min(image_width, int((canvas_width - self.preview_offset_x) / max(scale, 0.0001)) + 3)
        source_bottom = min(image_height, int((canvas_height - self.preview_offset_y) / max(scale, 0.0001)) + 3)
        if source_right <= source_left or source_bottom <= source_top:
            if redraw_minimap:
                self.redraw_minimap()
            return

        visible_crop = self.original_image.crop((source_left, source_top, source_right, source_bottom))
        visible_size = (
            max(1, int(round((source_right - source_left) * scale))),
            max(1, int(round((source_bottom - source_top) * scale))),
        )
        preview = visible_crop.resize(visible_size, Image.Resampling.LANCZOS)
        self.preview_image = preview
        self.preview_photo = ImageTk.PhotoImage(preview)
        crop_canvas_x = self.preview_offset_x + source_left * scale
        crop_canvas_y = self.preview_offset_y + source_top * scale

        self.canvas.create_image(
            crop_canvas_x,
            crop_canvas_y,
            image=self.preview_photo,
            anchor=tk.NW,
        )

        for index, spot in enumerate(self.spots):
            if self.spot_visible(spot):
                self._draw_spot(index, spot)

        self._draw_crop_rect()
        self._draw_lasso_path()
        if redraw_minimap:
            self.redraw_minimap()

    def spot_color(self, spot: Hotspot) -> tuple[str, str]:
        if self.spot_is_review(spot):
            return "#fecdd3", "#dc2626"
        if self.spot_is_inferred(spot):
            return "#c7d2fe", "#7c3aed"
        if spot.article:
            return "#ffe066", "#111827"
        return "#ffb86b", "#111827"

    def redraw_minimap(self) -> None:
        if not hasattr(self, "minimap"):
            return
        self.minimap.delete("all")
        if self.original_image is None:
            return

        canvas_width = max(self.minimap.winfo_width(), 1)
        canvas_height = max(self.minimap.winfo_height(), 1)
        image_width, image_height = self.original_image.size
        scale = min(canvas_width / image_width, canvas_height / image_height)
        mini_width = max(1, int(image_width * scale))
        mini_height = max(1, int(image_height * scale))
        offset_x = (canvas_width - mini_width) / 2
        offset_y = (canvas_height - mini_height) / 2

        thumbnail = self.original_image.resize((mini_width, mini_height), Image.Resampling.LANCZOS)
        self.minimap_photo = ImageTk.PhotoImage(thumbnail)
        self.minimap.create_image(offset_x, offset_y, image=self.minimap_photo, anchor=tk.NW)

        for index, spot in enumerate(self.spots):
            if not self.spot_visible(spot):
                continue
            _fill, outline = self.spot_color(spot)
            x = offset_x + spot.x * scale
            y = offset_y + spot.y * scale
            radius = 3 if index != self.selected_index else 5
            self.minimap.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=outline,
                outline="white" if index == self.selected_index else outline,
                width=2 if index == self.selected_index else 1,
            )

        if self.preview_scale > 0:
            left = max(0.0, (0 - self.preview_offset_x) / self.preview_scale)
            top = max(0.0, (0 - self.preview_offset_y) / self.preview_scale)
            right = min(float(image_width), (self.canvas.winfo_width() - self.preview_offset_x) / self.preview_scale)
            bottom = min(float(image_height), (self.canvas.winfo_height() - self.preview_offset_y) / self.preview_scale)
            self.minimap.create_rectangle(
                offset_x + left * scale,
                offset_y + top * scale,
                offset_x + right * scale,
                offset_y + bottom * scale,
                outline="#2563eb",
                width=2,
            )

    def on_minimap_click(self, event: tk.Event) -> None:
        if self.original_image is None:
            return
        canvas_width = max(self.minimap.winfo_width(), 1)
        canvas_height = max(self.minimap.winfo_height(), 1)
        image_width, image_height = self.original_image.size
        scale = min(canvas_width / image_width, canvas_height / image_height)
        mini_width = max(1, int(image_width * scale))
        mini_height = max(1, int(image_height * scale))
        offset_x = (canvas_width - mini_width) / 2
        offset_y = (canvas_height - mini_height) / 2
        image_x = (event.x - offset_x) / max(scale, 0.0001)
        image_y = (event.y - offset_y) / max(scale, 0.0001)
        if image_x < 0 or image_y < 0 or image_x > image_width or image_y > image_height:
            return
        self.center_view_on_point(image_x, image_y)
        self.redraw_canvas()

    def _draw_spot(self, index: int, spot: Hotspot) -> None:
        cx, cy = self.image_to_canvas(spot.x, spot.y)
        rect_w = max(spot.width * self.preview_scale, 18)
        rect_h = max(spot.height * self.preview_scale, 18)
        left = cx - rect_w / 2
        top = cy - rect_h / 2
        right = cx + rect_w / 2
        bottom = cy + rect_h / 2

        selected = index == self.selected_index
        fill, outline = self.spot_color(spot)
        if selected:
            outline = "#1d4ed8"
        width = 3 if selected else 1

        self.canvas.create_oval(left, top, right, bottom, fill=fill, outline=outline, width=width, tags=(f"spot-{index}",))
        self.canvas.create_text(
            cx,
            cy,
            text=spot.number,
            fill="#111827",
            font=("Segoe UI", max(8, int(10 * self.preview_scale + 6)), "bold"),
            tags=(f"spot-{index}",),
        )

    def _draw_crop_rect(self) -> None:
        if self.crop_rect is None:
            return

        x0, y0, x1, y1 = self.crop_rect
        left, top = self.image_to_canvas(x0, y0)
        right, bottom = self.image_to_canvas(x1, y1)
        self.canvas.create_rectangle(
            left,
            top,
            right,
            bottom,
            fill="#2563eb",
            stipple="gray25",
            outline="",
            tags=("crop-rect-fill",),
        )
        self.canvas.create_rectangle(
            left,
            top,
            right,
            bottom,
            outline="#2563eb",
            width=2,
            dash=(6, 4),
            tags=("crop-rect",),
        )

    def _draw_lasso_path(self) -> None:
        if not self.lasso_points:
            return

        points = [self.image_to_canvas(x, y) for x, y in self.lasso_points]
        flat_points = [coord for point in points for coord in point]
        if len(points) >= 3:
            self.canvas.create_polygon(
                *flat_points,
                fill="#16a34a",
                stipple="gray25",
                outline="",
                tags=("lasso-fill",),
            )
            self.canvas.create_line(
                *flat_points,
                points[0][0],
                points[0][1],
                fill="#16a34a",
                width=2,
                dash=(6, 4),
                tags=("lasso-outline",),
            )
        elif len(points) == 2:
            self.canvas.create_line(*flat_points, fill="#16a34a", width=2, tags=("lasso-outline",))

    def image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.preview_offset_x + x * self.preview_scale,
            self.preview_offset_y + y * self.preview_scale,
        )

    def canvas_to_image(self, x: float, y: float) -> tuple[float, float] | None:
        if self.original_image is None:
            return None
        ix = (x - self.preview_offset_x) / self.preview_scale
        iy = (y - self.preview_offset_y) / self.preview_scale
        if ix < 0 or iy < 0 or ix > self.original_image.width or iy > self.original_image.height:
            return None
        return ix, iy

    def nearest_spot_index(self, canvas_x: float, canvas_y: float) -> int | None:
        nearest: int | None = None
        nearest_distance = 999999.0
        for index, spot in enumerate(self.spots):
            if not self.spot_visible(spot):
                continue
            sx, sy = self.image_to_canvas(spot.x, spot.y)
            distance = ((sx - canvas_x) ** 2 + (sy - canvas_y) ** 2) ** 0.5
            radius = max(spot.width * self.preview_scale, spot.height * self.preview_scale, 18)
            if distance <= radius and distance < nearest_distance:
                nearest = index
                nearest_distance = distance
        return nearest

    def on_canvas_wheel(self, event: tk.Event) -> str:
        if self.original_image is None:
            return "break"

        zoom_in = False
        wheel_steps = 1.0
        if hasattr(event, "num") and event.num == 4:
            zoom_in = True
        elif hasattr(event, "num") and event.num == 5:
            zoom_in = False
        else:
            delta = int(getattr(event, "delta", 0) or 0)
            zoom_in = delta > 0
            wheel_steps = max(1.0, min(4.0, abs(delta) / 120))

        base_factor = 1.28
        factor = base_factor ** wheel_steps if zoom_in else base_factor ** (-wheel_steps)
        self.zoom_canvas_at(event.x, event.y, factor)
        return "break"

    def zoom_canvas_at(self, canvas_x: float, canvas_y: float, factor: float) -> None:
        if self.original_image is None:
            return

        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        image_width, image_height = self.original_image.size
        old_scale = max(self.preview_scale, 0.0001)
        image_x = (canvas_x - self.preview_offset_x) / old_scale
        image_y = (canvas_y - self.preview_offset_y) / old_scale
        image_x = min(max(image_x, 0.0), float(image_width))
        image_y = min(max(image_y, 0.0), float(image_height))

        fit_scale = self.fit_preview_scale(canvas_width, canvas_height)
        max_zoom = max(1.0, min(24.0, 10.0 / max(fit_scale, 0.0001)))
        new_zoom = min(max(self.viewport_zoom * factor, 1.0), max_zoom)
        if abs(new_zoom - self.viewport_zoom) < 0.001:
            return

        self.viewport_zoom = new_zoom
        new_scale = fit_scale * self.viewport_zoom
        preview_width = max(1, int(image_width * new_scale))
        preview_height = max(1, int(image_height * new_scale))
        base_x = (canvas_width - preview_width) / 2
        base_y = (canvas_height - preview_height) / 2
        self.viewport_pan_x = canvas_x - image_x * new_scale - base_x
        self.viewport_pan_y = canvas_y - image_y * new_scale - base_y
        self.clamp_viewport_pan(canvas_width, canvas_height, preview_width, preview_height)
        self.redraw_canvas()

    def start_canvas_pan(self, event: tk.Event) -> None:
        if self.original_image is None:
            return
        self.panning = True
        self.pan_start = (event.x, event.y)
        self.pan_origin = (self.viewport_pan_x, self.viewport_pan_y)
        self.canvas.configure(cursor="fleur")

    def on_canvas_down(self, event: tk.Event) -> None:
        if self.original_image is None:
            return

        image_point = self.canvas_to_image(event.x, event.y)

        if self.brush_mode.get():
            if image_point is None:
                return
            self.brush_dragging = True
            self.brush_last_point = image_point
            self.draw_brush_stroke(image_point, image_point, float(self.brush_size.get()))
            self.redraw_canvas(redraw_minimap=False)
            return

        if self.lasso_mode.get():
            if image_point is None:
                return
            self.lasso_points = [image_point]
            self.lasso_dragging = True
            self.crop_rect = None
            self.redraw_canvas(redraw_minimap=False)
            return

        if self.crop_mode.get():
            if image_point is None:
                return
            self.crop_start = image_point
            self.crop_rect = (*image_point, *image_point)
            self.crop_dragging = True
            self.redraw_canvas(redraw_minimap=False)
            return

        if self.add_mode.get():
            if image_point is None:
                return
            self.add_spot_at(*image_point)
            return

        index = self.nearest_spot_index(event.x, event.y)
        if index is not None:
            self.select_spot(index)
            self.dragging_index = index
            return

        self.start_canvas_pan(event)

    def on_canvas_drag(self, event: tk.Event) -> None:
        if self.brush_dragging and self.brush_last_point is not None:
            image_point = self.canvas_to_image(event.x, event.y)
            if image_point is None:
                return
            self.draw_brush_stroke(self.brush_last_point, image_point, float(self.brush_size.get()))
            self.brush_last_point = image_point
            self.redraw_canvas(redraw_minimap=False)
            return

        if self.lasso_dragging:
            image_point = self.canvas_to_image(event.x, event.y)
            if image_point is None:
                return
            if not self.lasso_points:
                self.lasso_points = [image_point]
            else:
                last_x, last_y = self.lasso_points[-1]
                if ((image_point[0] - last_x) ** 2 + (image_point[1] - last_y) ** 2) ** 0.5 < 2.0:
                    return
                self.lasso_points.append(image_point)
            self.redraw_canvas(redraw_minimap=False)
            return

        if self.crop_dragging and self.crop_start is not None:
            image_point = self.canvas_to_image(event.x, event.y)
            if image_point is None:
                return
            self.crop_rect = self.normalize_crop_rect(self.crop_start, image_point)
            self.redraw_canvas(redraw_minimap=False)
            return

        if self.dragging_index is not None:
            image_point = self.canvas_to_image(event.x, event.y)
            if image_point is None:
                return
            spot = self.spots[self.dragging_index]
            spot.x, spot.y = image_point
            self.redraw_canvas(redraw_minimap=False)
            return

        if self.panning and self.pan_start is not None and self.pan_origin is not None:
            canvas_width = max(self.canvas.winfo_width(), 1)
            canvas_height = max(self.canvas.winfo_height(), 1)
            preview_width = max(1, int(self.original_image.width * self.preview_scale))
            preview_height = max(1, int(self.original_image.height * self.preview_scale))
            self.viewport_pan_x = self.pan_origin[0] + event.x - self.pan_start[0]
            self.viewport_pan_y = self.pan_origin[1] + event.y - self.pan_start[1]
            self.clamp_viewport_pan(canvas_width, canvas_height, preview_width, preview_height)
            self.redraw_canvas(redraw_minimap=False)

    def on_canvas_up(self, _event: tk.Event) -> None:
        if self.brush_dragging:
            self.brush_dragging = False
            self.brush_last_point = None
            self.sync_work_image_file()
            self.redraw_canvas()
            self.status.set("Кисть применена. Можно запустить поиск заново.")
            return

        if self.lasso_dragging:
            self.lasso_dragging = False
            if not self.has_valid_lasso():
                self.lasso_points = []
                self.status.set("Контур лассо слишком маленький.")
            else:
                left, top, right, bottom = self.lasso_bounds()
                self.status.set(
                    f"Выделено лассо {int(right - left)}x{int(bottom - top)}. "
                    "Нажмите «Применить» или «Сохранить обрезанное»."
                )
            self.redraw_canvas()
            return

        if self.crop_dragging:
            self.crop_dragging = False
            self.crop_start = None
            if not self.has_valid_crop_rect():
                self.crop_rect = None
                self.status.set("Область обрезки слишком маленькая.")
            else:
                assert self.crop_rect is not None
                left, top, right, bottom = self.crop_rect
                self.status.set(
                    f"Выделена область {int(right - left)}x{int(bottom - top)}. "
                    "Нажмите «Применить» или «Сохранить обрезанное»."
                )
            self.redraw_canvas()
            return

        dragged_spot = self.dragging_index is not None
        if dragged_spot:
            self.refresh_tree()
        self.dragging_index = None
        if self.panning:
            self.panning = False
            self.pan_start = None
            self.pan_origin = None
            self.canvas.configure(cursor="")
            self.redraw_canvas()
            return
        if dragged_spot:
            self.redraw_canvas()

    def toggle_crop_mode(self) -> None:
        if self.crop_mode.get():
            self.add_mode.set(False)
            self.brush_mode.set(False)
            self.lasso_mode.set(False)
            self.lasso_dragging = False
            self.brush_dragging = False
            self.brush_last_point = None
            self.status.set("Режим обрезки: выделите область на картинке мышью.")

    def toggle_lasso_mode(self) -> None:
        if self.lasso_mode.get():
            self.add_mode.set(False)
            self.crop_mode.set(False)
            self.brush_mode.set(False)
            self.crop_dragging = False
            self.brush_dragging = False
            self.brush_last_point = None
            self.crop_rect = None
            self.status.set("Режим лассо: обведите область мышью, затем нажмите «Применить».")
        else:
            self.lasso_dragging = False

    def normalize_crop_rect(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> tuple[float, float, float, float]:
        if self.original_image is None:
            return 0.0, 0.0, 0.0, 0.0
        x0 = max(0.0, min(start[0], end[0]))
        y0 = max(0.0, min(start[1], end[1]))
        x1 = min(float(self.original_image.width), max(start[0], end[0]))
        y1 = min(float(self.original_image.height), max(start[1], end[1]))
        return x0, y0, x1, y1

    def has_valid_crop_rect(self) -> bool:
        if self.crop_rect is None:
            return False
        left, top, right, bottom = self.crop_rect
        return right - left >= 10 and bottom - top >= 10

    def has_valid_lasso(self) -> bool:
        if self.original_image is None or len(self.lasso_points) < 3:
            return False
        left, top, right, bottom = self.lasso_bounds()
        return right - left >= 10 and bottom - top >= 10

    def lasso_bounds(self) -> tuple[float, float, float, float]:
        if self.original_image is None or not self.lasso_points:
            return 0.0, 0.0, 0.0, 0.0
        xs = [point[0] for point in self.lasso_points]
        ys = [point[1] for point in self.lasso_points]
        return (
            max(0.0, min(xs)),
            max(0.0, min(ys)),
            min(float(self.original_image.width), max(xs)),
            min(float(self.original_image.height), max(ys)),
        )

    def lasso_box_pixels(self) -> tuple[int, int, int, int] | None:
        if self.original_image is None or not self.has_valid_lasso():
            return None
        left, top, right, bottom = self.lasso_bounds()
        return (
            max(0, int(round(left))),
            max(0, int(round(top))),
            min(self.original_image.width, int(round(right))),
            min(self.original_image.height, int(round(bottom))),
        )

    def crop_box_pixels(self) -> tuple[int, int, int, int] | None:
        if self.original_image is None or not self.has_valid_crop_rect():
            return None
        assert self.crop_rect is not None
        left, top, right, bottom = self.crop_rect
        return (
            max(0, int(round(left))),
            max(0, int(round(top))),
            min(self.original_image.width, int(round(right))),
            min(self.original_image.height, int(round(bottom))),
        )

    def clear_crop(self) -> None:
        self.crop_rect = None
        self.crop_start = None
        self.crop_dragging = False
        self.lasso_points = []
        self.lasso_dragging = False
        self.redraw_canvas()
        self.status.set("Область обрезки сброшена.")

    def rotate_left_90(self) -> None:
        self.rotate_current_image(90.0)

    def rotate_right_90(self) -> None:
        self.rotate_current_image(-90.0)

    def rotate_by_custom_angle(self) -> None:
        raw_angle = self.rotation_angle_var.get().strip().replace(",", ".")
        if not raw_angle:
            messagebox.showinfo("Нет угла", "Введите угол поворота в градусах.")
            return
        try:
            angle = float(raw_angle)
        except ValueError:
            messagebox.showerror("Неверный угол", "Введите число, например 7.5 или -12.")
            return
        self.rotate_current_image(angle)

    def rotate_point_on_canvas(
        self,
        x: float,
        y: float,
        angle: float,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        radians = math.radians(angle)
        cos_a = math.cos(radians)
        sin_a = math.sin(radians)
        cx = width / 2.0
        cy = height / 2.0
        dx = x - cx
        dy = y - cy
        return (
            cx + dx * cos_a + dy * sin_a,
            cy - dx * sin_a + dy * cos_a,
        )

    def rotate_axis_aligned_size(self, width: float, height: float, angle: float) -> tuple[float, float]:
        radians = math.radians(angle)
        cos_a = abs(math.cos(radians))
        sin_a = abs(math.sin(radians))
        return (
            max(1.0, width * cos_a + height * sin_a),
            max(1.0, width * sin_a + height * cos_a),
        )

    def rotate_image_in_place_canvas(self, image: Image.Image, angle: float, is_mask: bool = False) -> Image.Image:
        normalized = angle % 360.0
        if abs(normalized) < 0.0001 or abs(normalized - 360.0) < 0.0001:
            return image.copy()

        if abs(normalized - 90.0) < 0.0001:
            return image.transpose(Image.Transpose.ROTATE_90)
        if abs(normalized - 180.0) < 0.0001:
            return image.transpose(Image.Transpose.ROTATE_180)
        if abs(normalized - 270.0) < 0.0001:
            return image.transpose(Image.Transpose.ROTATE_270)

        resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
        fill = 0 if is_mask else "white"
        return image.rotate(angle, resample=resample, expand=False, fillcolor=fill)

    def rotate_cutout_regions(self, angle: float, image_size: tuple[int, int]) -> list[CutoutRegion]:
        rotated_regions: list[CutoutRegion] = []
        for region in self.cutout_regions:
            full_mask = Image.new("L", image_size, 0)
            if region.mask is None:
                ImageDraw.Draw(full_mask).rectangle(region.box, fill=255)
            else:
                mask = region.mask.convert("L")
                if mask.size != image_size:
                    mask = mask.resize(image_size, Image.Resampling.NEAREST)
                full_mask.paste(mask, (0, 0))

            rotated_mask = self.rotate_image_in_place_canvas(full_mask, angle, is_mask=True).convert("L")
            box = rotated_mask.getbbox()
            if box is None:
                continue
            rotated_regions.append(CutoutRegion(box=box, mask=rotated_mask))
        return rotated_regions

    def build_high_quality_current_canvas_for_rotation(self) -> Image.Image:
        saved_crop_rect = self.crop_rect
        saved_crop_start = self.crop_start
        saved_crop_dragging = self.crop_dragging
        saved_lasso_points = list(self.lasso_points)
        saved_lasso_dragging = self.lasso_dragging
        self.crop_rect = None
        self.crop_start = None
        self.crop_dragging = False
        self.lasso_points = []
        self.lasso_dragging = False
        try:
            image = self.build_high_quality_output_image()
            return self.fit_image_to_square(image, target_size=None)
        finally:
            self.crop_rect = saved_crop_rect
            self.crop_start = saved_crop_start
            self.crop_dragging = saved_crop_dragging
            self.lasso_points = saved_lasso_points
            self.lasso_dragging = saved_lasso_dragging

    def rotate_current_image(self, angle: float) -> None:
        if self.original_image is None:
            messagebox.showinfo("Нет изображения", "Сначала откройте изображение или PDF.")
            return

        normalized_angle = angle % 360.0
        if abs(normalized_angle) < 0.0001 or abs(normalized_angle - 360.0) < 0.0001:
            self.status.set("Поворот не выполнен: угол равен 0°.")
            return

        if not self.fragments:
            self.reset_fragments_from_current("Исходник")

        source_image = self.original_image.convert("RGB")
        width, height = source_image.size
        try:
            high_source = self.build_high_quality_current_canvas_for_rotation()
        except Exception:
            high_source = source_image.copy()
        rotated_high_source = self.rotate_image_in_place_canvas(high_source, angle).convert("RGB")
        rotated_image, high_to_work_scale, high_to_work_offset_x, high_to_work_offset_y = normalize_to_work_square(
            rotated_high_source
        )

        for spot in self.spots:
            spot.x, spot.y = self.rotate_point_on_canvas(spot.x, spot.y, angle, width, height)
            spot.x = min(max(spot.x, 0.0), float(rotated_image.width))
            spot.y = min(max(spot.y, 0.0), float(rotated_image.height))
            spot.width, spot.height = self.rotate_axis_aligned_size(spot.width, spot.height, angle)

        rotated_cutouts = self.rotate_cutout_regions(angle, (width, height))

        self.original_image = rotated_image
        self.raster_source_image = rotated_high_source
        self.source_image_size = rotated_high_source.size
        self.source_to_work_scale = high_to_work_scale
        self.source_to_work_offset_x = high_to_work_offset_x
        self.source_to_work_offset_y = high_to_work_offset_y
        self.pdf_source_to_work_scale = 1.0
        self.pdf_source_to_work_offset_x = 0.0
        self.pdf_source_to_work_offset_y = 0.0
        self.current_image_is_pdf_page = False
        self.source_type = "image"
        self.cutout_regions = rotated_cutouts
        self.brush_strokes = []
        self.image_before_brush = None
        self.brush_dragging = False
        self.brush_last_point = None
        self.crop_rect = None
        self.crop_start = None
        self.crop_dragging = False
        self.lasso_points = []
        self.lasso_dragging = False
        self.candidate_spots_by_number = {}
        self.current_candidate_options = []
        self.candidate_var.set("")
        self.ocr_run_id += 1
        self.leader_line_cache_key = None
        self.leader_line_cache = []
        self.line_score_cache = {}
        self.preview_image = None
        self.preview_photo = None
        self.image_path = self.write_temp_image(rotated_image, "rotated_current_800")
        self.reset_viewport()
        self.refresh_tree()
        self.redraw_canvas()
        self.save_current_fragment_state(refresh=True)
        self.status.set(f"Изображение повернуто на {angle:g}°. Рабочий формат сохранен: {rotated_image.width}x{rotated_image.height}.")

    def toggle_brush_mode(self) -> None:
        if self.brush_mode.get():
            self.crop_mode.set(False)
            self.lasso_mode.set(False)
            self.add_mode.set(False)
            self.lasso_dragging = False
            self.status.set("Кисть включена: зажмите левую кнопку мыши и закрасьте лишнее.")
        else:
            self.brush_dragging = False
            self.brush_last_point = None
            self.canvas.configure(cursor="")

    def sync_work_image_file(self, label: str = "painted_current_800") -> None:
        if self.original_image is None:
            return
        self.image_path = self.write_temp_image(self.original_image, label)
        self.leader_line_cache_key = None
        self.leader_line_cache = []

    def clear_brush_edits(self) -> None:
        if self.original_image is None:
            return
        if self.image_before_brush is None and not self.brush_strokes:
            self.status.set("Правок кистью нет.")
            return

        if self.image_before_brush is not None:
            self.original_image = self.image_before_brush.copy()
        self.brush_strokes = []
        self.image_before_brush = None
        self.brush_dragging = False
        self.brush_last_point = None
        self.sync_work_image_file("brush_reset_800")
        self.redraw_canvas()
        self.status.set("Правки кистью сброшены.")

    def draw_brush_stroke(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        size: float,
    ) -> None:
        if self.original_image is None:
            return
        if self.image_before_brush is None:
            self.image_before_brush = self.original_image.copy()

        radius = max(size / 2, 1.0)
        draw = ImageDraw.Draw(self.original_image)
        draw.line([start, end], fill="white", width=max(1, int(round(size))))
        for x, y in (start, end):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="white")
        self.brush_strokes.append((start[0], start[1], end[0], end[1], size))
        self.leader_line_cache_key = None
        self.leader_line_cache = []

    def apply_brush_strokes_to_ocr_image(
        self,
        image: Image.Image,
        coord_scale: float,
        offset_x: float,
        offset_y: float,
    ) -> Image.Image:
        if coord_scale <= 0:
            return image
        if not self.brush_strokes and not self.cutout_regions:
            return image

        result = image.copy()
        if self.cutout_regions:
            erase_mask = Image.new("L", result.size, 0)
            mask_draw = ImageDraw.Draw(erase_mask)
            for region in self.cutout_regions:
                left, top, right, bottom = region.box
                tx0 = (left - offset_x) / coord_scale
                ty0 = (top - offset_y) / coord_scale
                tx1 = (right - offset_x) / coord_scale
                ty1 = (bottom - offset_y) / coord_scale
                target_left = math.floor(min(tx0, tx1))
                target_top = math.floor(min(ty0, ty1))
                target_right = math.ceil(max(tx0, tx1))
                target_bottom = math.ceil(max(ty0, ty1))
                if target_right <= target_left or target_bottom <= target_top:
                    continue
                if target_right < 0 or target_bottom < 0 or target_left > result.width or target_top > result.height:
                    continue

                if region.mask is None:
                    mask_draw.rectangle((target_left, target_top, target_right, target_bottom), fill=255)
                    continue

                source_mask = region.mask.crop(region.box).convert("L")
                target_width = max(1, target_right - target_left)
                target_height = max(1, target_bottom - target_top)
                resized_mask = source_mask.resize((target_width, target_height), Image.Resampling.NEAREST)
                paste_left = max(0, target_left)
                paste_top = max(0, target_top)
                paste_right = min(result.width, target_right)
                paste_bottom = min(result.height, target_bottom)
                if paste_right <= paste_left or paste_bottom <= paste_top:
                    continue
                mask_crop = resized_mask.crop(
                    (
                        paste_left - target_left,
                        paste_top - target_top,
                        paste_right - target_left,
                        paste_bottom - target_top,
                    )
                )
                erase_mask.paste(mask_crop, (paste_left, paste_top))

            result.paste(Image.new("RGB", result.size, "white"), (0, 0), erase_mask)

        if not self.brush_strokes:
            return result

        draw = ImageDraw.Draw(result)
        for x0, y0, x1, y1, size in self.brush_strokes:
            sx0 = (x0 - offset_x) / coord_scale
            sy0 = (y0 - offset_y) / coord_scale
            sx1 = (x1 - offset_x) / coord_scale
            sy1 = (y1 - offset_y) / coord_scale
            stroke_width = max(1, int(round(size / coord_scale)))
            radius = stroke_width / 2
            draw.line([(sx0, sy0), (sx1, sy1)], fill="white", width=stroke_width)
            for x, y in ((sx0, sy0), (sx1, sy1)):
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="white")
        return result

    def distance_point_to_stroke(
        self,
        x: float,
        y: float,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> float:
        vx = x1 - x0
        vy = y1 - y0
        length_sq = vx * vx + vy * vy
        if length_sq <= 0.0001:
            return ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
        t = ((x - x0) * vx + (y - y0) * vy) / length_sq
        t = min(1.0, max(0.0, t))
        px = x0 + t * vx
        py = y0 + t * vy
        return ((x - px) ** 2 + (y - py) ** 2) ** 0.5

    def point_erased_by_brush(self, x: float, y: float) -> bool:
        for x0, y0, x1, y1, size in self.brush_strokes:
            if self.distance_point_to_stroke(x, y, x0, y0, x1, y1) <= size / 2 + 2:
                return True
        return False

    def distance_point_to_box(
        self,
        x: float,
        y: float,
        left: float,
        top: float,
        right: float,
        bottom: float,
    ) -> float:
        dx = max(left - x, 0.0, x - right)
        dy = max(top - y, 0.0, y - bottom)
        return (dx * dx + dy * dy) ** 0.5

    def spot_erased_by_brush(self, spot: Hotspot) -> bool:
        if not self.brush_strokes:
            return False

        left = spot.x - max(spot.width / 2, 4)
        right = spot.x + max(spot.width / 2, 4)
        top = spot.y - max(spot.height / 2, 4)
        bottom = spot.y + max(spot.height / 2, 4)

        for x0, y0, x1, y1, size in self.brush_strokes:
            radius = size / 2 + 3
            for step in range(6):
                ratio = step / 5
                x = x0 + (x1 - x0) * ratio
                y = y0 + (y1 - y0) * ratio
                if self.distance_point_to_box(x, y, left, top, right, bottom) <= radius:
                    return True
            if self.distance_point_to_stroke(spot.x, spot.y, x0, y0, x1, y1) <= radius:
                return True
        return False

    def build_lasso_mask(self) -> Image.Image | None:
        if self.original_image is None or not self.has_valid_lasso():
            return None
        mask = Image.new("L", self.original_image.size, 0)
        polygon = [
            (
                min(max(float(x), 0.0), float(self.original_image.width)),
                min(max(float(y), 0.0), float(self.original_image.height)),
            )
            for x, y in self.lasso_points
        ]
        ImageDraw.Draw(mask).polygon(polygon, fill=255)
        return mask

    def build_lasso_crop_image(
        self,
    ) -> tuple[Image.Image, tuple[int, int, int, int], Image.Image, float, float, float, float] | None:
        if self.original_image is None:
            return None
        box = self.lasso_box_pixels()
        mask = self.build_lasso_mask()
        if box is None or mask is None:
            return None

        left, top, right, bottom = box

        if self.current_image_is_pdf_page and self.pdf_document is not None and self.pdf_source_to_work_scale > 0:
            try:
                fitz = self.get_fitz(show_error=False)
                if fitz is not None:
                    page = self.pdf_document[self.pdf_page_index]
                    page_rect = page.rect
                    source_left = max(0.0, (left - self.pdf_source_to_work_offset_x) / self.pdf_source_to_work_scale)
                    source_top = max(0.0, (top - self.pdf_source_to_work_offset_y) / self.pdf_source_to_work_scale)
                    source_right = min(
                        page_rect.width * self.pdf_render_scale,
                        (right - self.pdf_source_to_work_offset_x) / self.pdf_source_to_work_scale,
                    )
                    source_bottom = min(
                        page_rect.height * self.pdf_render_scale,
                        (bottom - self.pdf_source_to_work_offset_y) / self.pdf_source_to_work_scale,
                    )
                    clip_rect = fitz.Rect(
                        page_rect.x0 + source_left / self.pdf_render_scale,
                        page_rect.y0 + source_top / self.pdf_render_scale,
                        page_rect.x0 + source_right / self.pdf_render_scale,
                        page_rect.y0 + source_bottom / self.pdf_render_scale,
                    )
                    if clip_rect.width > 1 and clip_rect.height > 1:
                        largest = max(clip_rect.width, clip_rect.height, 1)
                        target_scale = max(4.0, HQ_RENDER_TARGET_LONG_SIDE / largest)
                        target_scale = min(HQ_RENDER_MAX_SCALE, target_scale, HQ_MAX_SQUARE_SIZE / largest)
                        pixmap = page.get_pixmap(matrix=fitz.Matrix(target_scale, target_scale), alpha=False, clip=clip_rect)
                        source_crop = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                        sx = target_scale / (self.pdf_render_scale * self.pdf_source_to_work_scale)
                        sy = sx
                        dx = (
                            -self.pdf_source_to_work_offset_x / self.pdf_source_to_work_scale - source_left
                        ) * target_scale / self.pdf_render_scale
                        dy = (
                            -self.pdf_source_to_work_offset_y / self.pdf_source_to_work_scale - source_top
                        ) * target_scale / self.pdf_render_scale
                        source_crop = self.apply_brush_strokes_to_ocr_image(
                            source_crop,
                            1.0 / sx,
                            -dx / sx,
                            -dy / sy,
                        )
                        polygon = [(x * sx + dx, y * sy + dy) for x, y in self.lasso_points]
                        high_mask = Image.new("L", source_crop.size, 0)
                        ImageDraw.Draw(high_mask).polygon(polygon, fill=255)
                        result = Image.new("RGB", source_crop.size, "white")
                        result.paste(source_crop, (0, 0), high_mask)
                        return result, box, mask, sx, sy, dx, dy
            except Exception:
                pass

        if self.raster_source_image is not None and self.source_to_work_scale > 0:
            try:
                source_points = [
                    (
                        (x - self.source_to_work_offset_x) / self.source_to_work_scale,
                        (y - self.source_to_work_offset_y) / self.source_to_work_scale,
                    )
                    for x, y in self.lasso_points
                ]
                xs = [point[0] for point in source_points]
                ys = [point[1] for point in source_points]
                source_box = (
                    max(0, int(round(min(xs)))),
                    max(0, int(round(min(ys)))),
                    min(self.raster_source_image.width, int(round(max(xs)))),
                    min(self.raster_source_image.height, int(round(max(ys)))),
                )
                if source_box[2] > source_box[0] and source_box[3] > source_box[1]:
                    source_crop = self.raster_source_image.crop(source_box).convert("RGB")
                    sx = 1.0 / self.source_to_work_scale
                    sy = sx
                    dx = -self.source_to_work_offset_x / self.source_to_work_scale - source_box[0]
                    dy = -self.source_to_work_offset_y / self.source_to_work_scale - source_box[1]
                    source_crop = self.apply_brush_strokes_to_ocr_image(
                        source_crop,
                        1.0 / sx,
                        -dx / sx,
                        -dy / sy,
                    )
                    polygon = [(x * sx + dx, y * sy + dy) for x, y in self.lasso_points]
                    high_mask = Image.new("L", source_crop.size, 0)
                    ImageDraw.Draw(high_mask).polygon(polygon, fill=255)
                    result = Image.new("RGB", source_crop.size, "white")
                    result.paste(source_crop, (0, 0), high_mask)
                    return result, box, mask, sx, sy, dx, dy
            except Exception:
                pass

        source_crop = self.original_image.crop(box).convert("RGB")
        source_crop = self.apply_brush_strokes_to_ocr_image(source_crop, 1.0, float(left), float(top))
        mask_crop = mask.crop(box)
        result = Image.new("RGB", source_crop.size, "white")
        result.paste(source_crop, (0, 0), mask_crop)
        return result, box, mask, 1.0, 1.0, -float(left), -float(top)

    def build_lasso_fragment(self, name: str) -> ImageFragment | None:
        lasso_crop = self.build_lasso_crop_image()
        if lasso_crop is None:
            return None

        cropped, _box, mask, crop_sx, crop_sy, crop_dx, crop_dy = lasso_crop
        source_spots: list[Hotspot] = []
        for original_spot in self.spots:
            px = min(max(int(round(original_spot.x)), 0), mask.width - 1)
            py = min(max(int(round(original_spot.y)), 0), mask.height - 1)
            if mask.getpixel((px, py)) <= 0:
                continue
            spot = Hotspot(**asdict(original_spot))
            spot.x = spot.x * crop_sx + crop_dx
            spot.y = spot.y * crop_sy + crop_dy
            spot.width *= crop_sx
            spot.height *= crop_sy
            source_spots.append(spot)

        normalized, kept_spots, quality_source, crop_scale, crop_offset_x, crop_offset_y = self.finalize_quality_800_image(
            cropped,
            source_spots,
        )

        return ImageFragment(
            name=name,
            image=normalized,
            image_path=self.write_temp_image(normalized, f"fragment_lasso_{len(self.fragments) + 1}_800"),
            source_image_size=quality_source.size,
            source_to_work_scale=crop_scale,
            source_to_work_offset_x=crop_offset_x,
            source_to_work_offset_y=crop_offset_y,
            pdf_source_to_work_scale=1.0,
            pdf_source_to_work_offset_x=0.0,
            pdf_source_to_work_offset_y=0.0,
            current_image_is_pdf_page=False,
            raster_source_image=quality_source.copy(),
            spots=kept_spots,
            brush_strokes=[],
            cutout_regions=[],
        )

    def build_rect_fragment(self, name: str) -> ImageFragment | None:
        if self.original_image is None:
            return None
        box = self.crop_box_pixels()
        if box is None:
            return None

        left, top, right, bottom = box
        keep_pdf_ocr_mode = self.source_type == "pdf" and self.pdf_document is not None
        raster_crop_image: Image.Image | None = None
        raster_source_box: tuple[int, int, int, int] | None = None
        if not keep_pdf_ocr_mode and self.raster_source_image is not None:
            source_left, source_top = self.work_to_source_point(left, top)
            source_right, source_bottom = self.work_to_source_point(right, bottom)
            source_box = (
                max(0, int(round(min(source_left, source_right)))),
                max(0, int(round(min(source_top, source_bottom)))),
                min(self.raster_source_image.width, int(round(max(source_left, source_right)))),
                min(self.raster_source_image.height, int(round(max(source_top, source_bottom)))),
            )
            if source_box[2] > source_box[0] and source_box[3] > source_box[1]:
                raster_source_box = source_box
                raster_crop_image = self.raster_source_image.crop(source_box).convert("RGB")
                raster_crop_image = self.apply_brush_strokes_to_ocr_image(
                    raster_crop_image,
                    self.source_to_work_scale,
                    self.source_to_work_offset_x + source_box[0] * self.source_to_work_scale,
                    self.source_to_work_offset_y + source_box[1] * self.source_to_work_scale,
                )

        work_cropped = self.original_image.crop(box).convert("RGB")
        work_cropped = self.apply_brush_strokes_to_ocr_image(work_cropped, 1.0, float(left), float(top))
        cropped = work_cropped
        if keep_pdf_ocr_mode:
            try:
                cropped = self.render_high_quality_pdf_png()
            except Exception:
                cropped = work_cropped
        elif raster_crop_image is not None:
            cropped = raster_crop_image

        old_source_scale = self.source_to_work_scale
        old_source_offset_x = self.source_to_work_offset_x
        old_source_offset_y = self.source_to_work_offset_y
        source_spots: list[Hotspot] = []
        crop_width = max(right - left, 1)
        crop_height = max(bottom - top, 1)
        for original_spot in self.spots:
            if left <= original_spot.x <= right and top <= original_spot.y <= bottom:
                spot = Hotspot(**asdict(original_spot))
                if raster_source_box is not None and old_source_scale > 0:
                    source_x = (spot.x - old_source_offset_x) / old_source_scale
                    source_y = (spot.y - old_source_offset_y) / old_source_scale
                    spot.x = source_x - raster_source_box[0]
                    spot.y = source_y - raster_source_box[1]
                    spot.width = max(1.0, spot.width / old_source_scale)
                    spot.height = max(1.0, spot.height / old_source_scale)
                else:
                    spot.x = (spot.x - left) / crop_width * cropped.width
                    spot.y = (spot.y - top) / crop_height * cropped.height
                    spot.width = max(1.0, spot.width / crop_width * cropped.width)
                    spot.height = max(1.0, spot.height / crop_height * cropped.height)
                source_spots.append(spot)

        normalized, kept_spots, quality_source, source_to_work_scale, source_to_work_offset_x, source_to_work_offset_y = (
            self.finalize_quality_800_image(cropped, source_spots)
        )
        transformed_strokes: list[tuple[float, float, float, float, float]] = []
        pdf_source_to_work_scale = 1.0
        pdf_source_to_work_offset_x = 0.0
        pdf_source_to_work_offset_y = 0.0
        current_image_is_pdf_page = False
        raster_source_for_fragment = quality_source.copy()

        return ImageFragment(
            name=name,
            image=normalized,
            image_path=self.write_temp_image(normalized, f"fragment_rect_{len(self.fragments) + 1}_800"),
            source_image_size=quality_source.size,
            source_to_work_scale=source_to_work_scale,
            source_to_work_offset_x=source_to_work_offset_x,
            source_to_work_offset_y=source_to_work_offset_y,
            pdf_source_to_work_scale=pdf_source_to_work_scale,
            pdf_source_to_work_offset_x=pdf_source_to_work_offset_x,
            pdf_source_to_work_offset_y=pdf_source_to_work_offset_y,
            current_image_is_pdf_page=current_image_is_pdf_page,
            raster_source_image=raster_source_for_fragment,
            spots=kept_spots,
            brush_strokes=transformed_strokes,
            cutout_regions=[],
        )

    def add_erasure_strokes_for_box(self, left: float, top: float, right: float, bottom: float) -> None:
        stroke_size = 22.0
        step = max(8.0, stroke_size * 0.65)
        y = top
        while y <= bottom:
            self.brush_strokes.append((left, y, right, y, stroke_size))
            y += step
        self.brush_strokes.append((left, bottom, right, bottom, stroke_size))

    def current_selection_box_and_mask(self) -> tuple[tuple[int, int, int, int], Image.Image | None] | None:
        if self.has_valid_lasso():
            box = self.lasso_box_pixels()
            mask = self.build_lasso_mask()
            if box is None or mask is None:
                return None
            return box, mask

        box = self.crop_box_pixels()
        if box is None:
            return None
        return box, None

    def nonzero_mask_pixels(self, mask: Image.Image, box: tuple[int, int, int, int]) -> int:
        crop = mask.crop(box).convert("L")
        histogram = crop.histogram()
        return sum(histogram[1:])

    def overlapping_mask_pixels(
        self,
        first: Image.Image,
        second: Image.Image,
        box: tuple[int, int, int, int],
    ) -> int:
        first_crop = first.crop(box).convert("L")
        second_crop = second.crop(box).convert("L")
        first_data = first_crop.tobytes()
        second_data = second_crop.tobytes()
        return sum(1 for left, right in zip(first_data, second_data) if left and right)

    def selection_overlaps_existing_cutout(self) -> bool:
        selection = self.current_selection_box_and_mask()
        if selection is None or not self.cutout_regions:
            return False

        selection_box, selection_mask = selection
        sel_left, sel_top, sel_right, sel_bottom = selection_box
        min_overlap_pixels = 40
        for region in self.cutout_regions:
            cut_left, cut_top, cut_right, cut_bottom = region.box
            left = max(sel_left, cut_left)
            top = max(sel_top, cut_top)
            right = min(sel_right, cut_right)
            bottom = min(sel_bottom, cut_bottom)
            if right <= left or bottom <= top:
                continue

            intersection = (left, top, right, bottom)
            if selection_mask is None and region.mask is None:
                overlap_pixels = (right - left) * (bottom - top)
            elif selection_mask is None and region.mask is not None:
                overlap_pixels = self.nonzero_mask_pixels(region.mask, intersection)
            elif selection_mask is not None and region.mask is None:
                overlap_pixels = self.nonzero_mask_pixels(selection_mask, intersection)
            else:
                assert selection_mask is not None and region.mask is not None
                overlap_pixels = self.overlapping_mask_pixels(selection_mask, region.mask, intersection)

            if overlap_pixels >= min_overlap_pixels:
                return True
        return False

    def remove_spots_in_erased_area(
        self,
        box: tuple[int, int, int, int],
        mask: Image.Image | None = None,
    ) -> None:
        left, top, right, bottom = box
        kept: list[Hotspot] = []
        for spot in self.spots:
            if not (left <= spot.x <= right and top <= spot.y <= bottom):
                kept.append(spot)
                continue
            if mask is not None:
                px = min(max(int(round(spot.x)), 0), mask.width - 1)
                py = min(max(int(round(spot.y)), 0), mask.height - 1)
                if mask.getpixel((px, py)) <= 0:
                    kept.append(spot)
                    continue
            # Drop spots inside the cut-out area on the source/current fragment.
        self.spots = kept

    def erase_selection_from_current_image(self) -> bool:
        if self.original_image is None:
            return False

        draw = ImageDraw.Draw(self.original_image)
        erased = False
        if self.has_valid_lasso():
            box = self.lasso_box_pixels()
            mask = self.build_lasso_mask()
            if box is None or mask is None:
                return False
            polygon = [
                (
                    min(max(float(x), 0.0), float(self.original_image.width)),
                    min(max(float(y), 0.0), float(self.original_image.height)),
                )
                for x, y in self.lasso_points
            ]
            draw.polygon(polygon, fill="white")
            left, top, right, bottom = box
            self.add_erasure_strokes_for_box(float(left), float(top), float(right), float(bottom))
            self.remove_spots_in_erased_area(box, mask)
            self.cutout_regions.append(CutoutRegion(box=box, mask=mask.copy()))
            if self.raster_source_image is not None and self.source_to_work_scale > 0:
                source_polygon = [
                    (
                        (x - self.source_to_work_offset_x) / self.source_to_work_scale,
                        (y - self.source_to_work_offset_y) / self.source_to_work_scale,
                    )
                    for x, y in self.lasso_points
                ]
                ImageDraw.Draw(self.raster_source_image).polygon(source_polygon, fill="white")
            erased = True
        else:
            box = self.crop_box_pixels()
            if box is None:
                return False
            left, top, right, bottom = box
            draw.rectangle((left, top, right, bottom), fill="white")
            self.add_erasure_strokes_for_box(float(left), float(top), float(right), float(bottom))
            self.remove_spots_in_erased_area(box)
            self.cutout_regions.append(CutoutRegion(box=box, mask=None))
            if self.raster_source_image is not None and self.source_to_work_scale > 0:
                source_left, source_top = self.work_to_source_point(left, top)
                source_right, source_bottom = self.work_to_source_point(right, bottom)
                ImageDraw.Draw(self.raster_source_image).rectangle(
                    (
                        min(source_left, source_right),
                        min(source_top, source_bottom),
                        max(source_left, source_right),
                        max(source_top, source_bottom),
                    ),
                    fill="white",
                )
            erased = True

        if erased:
            self.image_before_brush = None
            self.leader_line_cache_key = None
            self.leader_line_cache = []
            self.line_score_cache = {}
            self.preview_image = None
            self.preview_photo = None
            self.sync_work_image_file("fragment_source_erased_800")
        return erased

    def fragment_has_visible_content(self, fragment: ImageFragment) -> bool:
        gray = fragment.image.convert("L")
        histogram = gray.histogram()
        dark_pixels = sum(count for value, count in enumerate(histogram) if value < 245)
        return dark_pixels >= 30

    def add_fragment_from_selection(self, switch_to_new: bool = False) -> None:
        if self.original_image is None:
            messagebox.showinfo("Нет изображения", "Сначала откройте изображение или PDF.")
            return
        if not self.fragments:
            self.reset_fragments_from_current("Исходник")

        if self.current_selection_box_and_mask() is None:
            messagebox.showinfo(
                "Нет области",
                "Выделите прямоугольник или обведите область лассо, затем создайте фрагмент.",
            )
            return
        overlaps_existing = self.selection_overlaps_existing_cutout()

        name = self.next_fragment_name()
        if self.has_valid_lasso():
            fragment = self.build_lasso_fragment(name)
        else:
            fragment = self.build_rect_fragment(name)
        if fragment is None:
            messagebox.showinfo(
                "Нет области",
                "Выделите прямоугольник или обведите область лассо, затем нажмите «Добавить фрагмент справа».",
            )
            return

        if not self.fragment_has_visible_content(fragment):
            messagebox.showinfo(
                "Пустой фрагмент",
                "В выбранной области почти нет видимого содержимого. Возможно, этот участок уже был вырезан.",
            )
            return

        self.erase_selection_from_current_image()
        self.save_current_fragment_state()
        self.fragments.append(fragment)
        new_index = len(self.fragments) - 1
        self.crop_rect = None
        self.crop_start = None
        self.crop_dragging = False
        self.lasso_points = []
        self.lasso_dragging = False
        if switch_to_new:
            self.switch_fragment(new_index, save_current=False)
            overlap_note = " Уже вырезанная часть внутри него не взята." if overlaps_existing else ""
            self.status.set(f"{name} добавлен справа и открыт. Для возврата нажмите «Исходник» в правой колонке.{overlap_note}")
            return
        self.refresh_fragment_buttons()
        self.redraw_canvas()
        overlap_note = " Уже вырезанная часть внутри него не взята." if overlaps_existing else ""
        self.status.set(f"{name} добавлен справа. Можно выделять следующую область или открыть фрагмент кнопкой.{overlap_note}")

    def apply_lasso_crop(self) -> None:
        if self.original_image is None:
            return
        lasso_crop = self.build_lasso_crop_image()
        if lasso_crop is None:
            messagebox.showinfo("Нет области", "Включите режим лассо и обведите область мышью.")
            return

        cropped, box, mask, crop_sx, crop_sy, crop_dx, crop_dy = lasso_crop
        left, top, right, bottom = box
        normalized, crop_scale, crop_offset_x, crop_offset_y = normalize_to_work_square(cropped)
        kept_spots: list[Hotspot] = []
        for spot in self.spots:
            px = min(max(int(round(spot.x)), 0), mask.width - 1)
            py = min(max(int(round(spot.y)), 0), mask.height - 1)
            if mask.getpixel((px, py)) <= 0:
                continue
            spot.x = (spot.x * crop_sx + crop_dx) * crop_scale + crop_offset_x
            spot.y = (spot.y * crop_sy + crop_dy) * crop_scale + crop_offset_y
            spot.width *= crop_sx * crop_scale
            spot.height *= crop_sy * crop_scale
            kept_spots.append(spot)

        self.original_image = normalized
        self.source_image_size = normalized.size
        self.source_to_work_scale = 1.0
        self.source_to_work_offset_x = 0.0
        self.source_to_work_offset_y = 0.0
        self.pdf_source_to_work_scale = 1.0
        self.pdf_source_to_work_offset_x = 0.0
        self.pdf_source_to_work_offset_y = 0.0
        self.current_image_is_pdf_page = False
        self.raster_source_image = normalized.copy()
        self.brush_strokes = []
        self.image_before_brush = None
        self.brush_dragging = False
        self.brush_last_point = None
        self.image_path = self.write_temp_image(normalized, "lasso_current_800")
        self.preview_image = None
        self.preview_photo = None
        self.spots = kept_spots
        self.selected_index = None
        self.crop_rect = None
        self.crop_start = None
        self.crop_dragging = False
        self.lasso_points = []
        self.lasso_dragging = False
        self.reset_viewport()
        self.refresh_tree()
        self.redraw_canvas()
        self.status.set(
            f"Лассо применено: область {cropped.width}x{cropped.height}, рабочий формат {WORK_IMAGE_SIZE}x{WORK_IMAGE_SIZE}, "
            f"цифр внутри области {len(kept_spots)}."
        )

    def apply_crop(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("Нет изображения", "Сначала откройте изображение или PDF.")
            return

        self.add_fragment_from_selection(switch_to_new=True)
        return

        if self.has_valid_lasso():
            self.apply_lasso_crop()
            return

        box = self.crop_box_pixels()
        if box is None:
            messagebox.showinfo("Нет области", "Включите режим обрезки и выделите область мышью.")
            return

        left, top, right, bottom = box
        keep_pdf_ocr_mode = self.source_type == "pdf" and self.pdf_document is not None
        raster_crop_image: Image.Image | None = None
        raster_source_box: tuple[int, int, int, int] | None = None
        if not keep_pdf_ocr_mode and self.raster_source_image is not None:
            source_left, source_top = self.work_to_source_point(left, top)
            source_right, source_bottom = self.work_to_source_point(right, bottom)
            source_box = (
                max(0, int(round(min(source_left, source_right)))),
                max(0, int(round(min(source_top, source_bottom)))),
                min(self.raster_source_image.width, int(round(max(source_left, source_right)))),
                min(self.raster_source_image.height, int(round(max(source_top, source_bottom)))),
            )
            if source_box[2] > source_box[0] and source_box[3] > source_box[1]:
                raster_source_box = source_box
                raster_crop_image = self.raster_source_image.crop(source_box).convert("RGB")

        work_cropped = self.original_image.crop(box).convert("RGB")
        cropped = work_cropped
        if keep_pdf_ocr_mode:
            try:
                cropped = self.render_high_quality_pdf_png()
            except Exception:
                cropped = work_cropped
        elif raster_crop_image is not None:
            cropped = raster_crop_image

        normalized, visual_crop_scale, visual_crop_offset_x, visual_crop_offset_y = normalize_to_work_square(cropped)
        work_crop_scale = min(WORK_IMAGE_SIZE / max(right - left, 1), WORK_IMAGE_SIZE / max(bottom - top, 1))
        work_crop_offset_x = (WORK_IMAGE_SIZE - (right - left) * work_crop_scale) / 2
        work_crop_offset_y = (WORK_IMAGE_SIZE - (bottom - top) * work_crop_scale) / 2
        old_pdf_scale = self.pdf_source_to_work_scale
        old_pdf_offset_x = self.pdf_source_to_work_offset_x
        old_pdf_offset_y = self.pdf_source_to_work_offset_y
        old_source_scale = self.source_to_work_scale
        old_source_offset_x = self.source_to_work_offset_x
        old_source_offset_y = self.source_to_work_offset_y
        old_brush_strokes = list(self.brush_strokes)

        def transform_work_point(x: float, y: float) -> tuple[float, float]:
            if raster_source_box is not None and old_source_scale > 0:
                source_x = (x - old_source_offset_x) / old_source_scale
                source_y = (y - old_source_offset_y) / old_source_scale
                return (
                    (source_x - raster_source_box[0]) * visual_crop_scale + visual_crop_offset_x,
                    (source_y - raster_source_box[1]) * visual_crop_scale + visual_crop_offset_y,
                )
            return (
                (x - left) * work_crop_scale + work_crop_offset_x,
                (y - top) * work_crop_scale + work_crop_offset_y,
            )

        def transform_work_size(width: float, height: float) -> tuple[float, float]:
            if raster_source_box is not None and old_source_scale > 0:
                factor = visual_crop_scale / old_source_scale
                return width * factor, height * factor
            return width * work_crop_scale, height * work_crop_scale

        kept_spots: list[Hotspot] = []
        for spot in self.spots:
            if left <= spot.x <= right and top <= spot.y <= bottom:
                spot.x, spot.y = transform_work_point(spot.x, spot.y)
                spot.width, spot.height = transform_work_size(spot.width, spot.height)
                kept_spots.append(spot)

        self.original_image = normalized
        self.source_image_size = cropped.size
        self.source_to_work_scale = visual_crop_scale if raster_source_box is not None else work_crop_scale
        self.source_to_work_offset_x = visual_crop_offset_x if raster_source_box is not None else work_crop_offset_x
        self.source_to_work_offset_y = visual_crop_offset_y if raster_source_box is not None else work_crop_offset_y
        if keep_pdf_ocr_mode:
            self.pdf_source_to_work_scale = old_pdf_scale * work_crop_scale
            self.pdf_source_to_work_offset_x = (old_pdf_offset_x - left) * work_crop_scale + work_crop_offset_x
            self.pdf_source_to_work_offset_y = (old_pdf_offset_y - top) * work_crop_scale + work_crop_offset_y
            self.current_image_is_pdf_page = True
        else:
            self.pdf_source_to_work_scale = 1.0
            self.pdf_source_to_work_offset_x = 0.0
            self.pdf_source_to_work_offset_y = 0.0
            self.current_image_is_pdf_page = False
            if raster_crop_image is not None:
                self.raster_source_image = raster_crop_image
            else:
                self.raster_source_image = normalized.copy()
        transformed_strokes: list[tuple[float, float, float, float, float]] = []
        for x0, y0, x1, y1, size in old_brush_strokes:
            if max(x0, x1) < left or min(x0, x1) > right or max(y0, y1) < top or min(y0, y1) > bottom:
                continue
            tx0, ty0 = transform_work_point(x0, y0)
            tx1, ty1 = transform_work_point(x1, y1)
            tw, _th = transform_work_size(size, size)
            transformed_strokes.append(
                (
                    tx0,
                    ty0,
                    tx1,
                    ty1,
                    max(1.0, tw),
                )
            )
        self.brush_strokes = transformed_strokes
        self.image_before_brush = None
        self.brush_dragging = False
        self.brush_last_point = None
        self.image_path = self.write_temp_image(normalized, "cropped_current_800")
        self.preview_image = None
        self.preview_photo = None
        self.spots = kept_spots
        self.selected_index = None
        self.crop_rect = None
        self.crop_start = None
        self.lasso_points = []
        self.lasso_dragging = False
        self.reset_viewport()
        self.refresh_tree()
        self.redraw_canvas()
        self.status.set(
            f"Обрезка применена: область {cropped.width}x{cropped.height}, рабочий формат {WORK_IMAGE_SIZE}x{WORK_IMAGE_SIZE}, "
            f"цифр внутри области {len(kept_spots)}."
        )

    def save_cropped_file(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("Нет изображения", "Сначала откройте изображение или PDF.")
            return

        self.save_cropped_image()

    def save_cropped_image(self) -> None:
        assert self.original_image is not None
        default_ext = ".png"
        file_name = filedialog.asksaveasfilename(
            title="Сохранить изображение в высоком качестве",
            defaultextension=default_ext,
            filetypes=(
                ("PNG", "*.png"),
                ("JPEG", "*.jpg;*.jpeg"),
                ("WebP", "*.webp"),
                ("All files", "*.*"),
            ),
        )
        if not file_name:
            return

        try:
            image = self.build_high_quality_output_image()
            image = self.fit_image_to_final_800_square(image)
        except Exception as exc:
            messagebox.showerror("Ошибка сохранения", f"Не удалось подготовить изображение в высоком качестве:\n{exc}")
            return

        suffix = Path(file_name).suffix.lower()
        save_kwargs = {}
        if suffix in {".jpg", ".jpeg"}:
            save_kwargs = {"quality": 100, "subsampling": 0, "optimize": True}
        elif suffix == ".webp":
            save_kwargs = {"quality": 100, "lossless": True, "method": 6}
        elif suffix == ".png":
            save_kwargs = {"compress_level": 1, "dpi": (300, 300)}
        image.save(file_name, **save_kwargs)
        self.status.set(f"Изображение сохранено в формате 800x800 из high-res источника: {file_name}")

    def save_cropped_pdf(self) -> None:
        if self.source_path is None or self.original_image is None or self.crop_rect is None:
            return

        fitz = self.get_fitz()
        if fitz is None:
            return

        file_name = filedialog.asksaveasfilename(
            title="Сохранить обрезанный PDF",
            defaultextension=".pdf",
            filetypes=(("PDF", "*.pdf"), ("All files", "*.*")),
        )
        if not file_name:
            return

        if Path(file_name).resolve() == self.source_path.resolve():
            messagebox.showwarning(
                "Выберите новый файл",
                "Обрезанный PDF нужно сохранить в новый файл, не поверх исходного.",
            )
            return

        left, top, right, bottom = self.crop_rect
        source_left, source_top = self.work_to_source_point(left, top)
        source_right, source_bottom = self.work_to_source_point(right, bottom)
        source_width, source_height = self.source_image_size
        source_left = max(0.0, min(source_left, float(source_width)))
        source_right = max(0.0, min(source_right, float(source_width)))
        source_top = max(0.0, min(source_top, float(source_height)))
        source_bottom = max(0.0, min(source_bottom, float(source_height)))
        if source_right <= source_left or source_bottom <= source_top:
            messagebox.showwarning("Нет области", "Выделенная область попала за пределы страницы PDF.")
            return

        rel = (
            source_left / max(source_width, 1),
            source_top / max(source_height, 1),
            source_right / max(source_width, 1),
            source_bottom / max(source_height, 1),
        )

        try:
            document = fitz.open(self.source_path)
            page_indexes = (
                range(document.page_count)
                if self.crop_all_pdf_pages.get()
                else [self.pdf_page_index]
            )
            cropped_pages = 0
            for page_index in page_indexes:
                page = document[page_index]
                base = page.cropbox
                new_box = fitz.Rect(
                    base.x0 + base.width * rel[0],
                    base.y0 + base.height * rel[1],
                    base.x0 + base.width * rel[2],
                    base.y0 + base.height * rel[3],
                )
                if new_box.width < 5 or new_box.height < 5:
                    continue
                page.set_cropbox(new_box)
                cropped_pages += 1
            document.save(file_name, garbage=4, deflate=True)
            document.close()
        except Exception as exc:
            messagebox.showerror("Ошибка PDF", f"Не удалось сохранить обрезанный PDF:\n{exc}")
            return

        mode = "на всех страницах" if self.crop_all_pdf_pages.get() else "на текущей странице"
        self.status.set(f"Обрезанный PDF сохранён ({mode}, страниц: {cropped_pages}): {file_name}")

    def current_pdf_clip_rect(self):
        if self.pdf_document is None or self.original_image is None:
            return None
        fitz = self.get_fitz(show_error=False)
        if fitz is None:
            return None

        page = self.pdf_document[self.pdf_page_index]
        page_rect = page.rect
        if self.has_valid_crop_rect():
            work_left, work_top, work_right, work_bottom = self.crop_box_pixels() or (
                0,
                0,
                self.original_image.width,
                self.original_image.height,
            )
        else:
            work_left, work_top, work_right, work_bottom = (
                0,
                0,
                self.original_image.width,
                self.original_image.height,
            )

        if self.pdf_source_to_work_scale <= 0:
            return page_rect

        source_left = max(0.0, (work_left - self.pdf_source_to_work_offset_x) / self.pdf_source_to_work_scale)
        source_top = max(0.0, (work_top - self.pdf_source_to_work_offset_y) / self.pdf_source_to_work_scale)
        source_right = min(
            page_rect.width * self.pdf_render_scale,
            (work_right - self.pdf_source_to_work_offset_x) / self.pdf_source_to_work_scale,
        )
        source_bottom = min(
            page_rect.height * self.pdf_render_scale,
            (work_bottom - self.pdf_source_to_work_offset_y) / self.pdf_source_to_work_scale,
        )
        if source_right <= source_left + 1 or source_bottom <= source_top + 1:
            return page_rect

        return fitz.Rect(
            page_rect.x0 + source_left / self.pdf_render_scale,
            page_rect.y0 + source_top / self.pdf_render_scale,
            page_rect.x0 + source_right / self.pdf_render_scale,
            page_rect.y0 + source_bottom / self.pdf_render_scale,
        )

    def render_high_quality_pdf_png(self) -> Image.Image:
        if self.pdf_document is None or self.original_image is None:
            raise OcrError("PDF не открыт")
        fitz = self.get_fitz()
        if fitz is None:
            raise OcrError("PyMuPDF недоступен")

        page = self.pdf_document[self.pdf_page_index]
        page_rect = page.rect
        clip_rect = self.current_pdf_clip_rect() or page_rect
        largest = max(clip_rect.width, clip_rect.height, 1)
        target_scale = max(4.0, HQ_RENDER_TARGET_LONG_SIDE / largest)
        target_scale = min(HQ_RENDER_MAX_SCALE, target_scale, HQ_MAX_SQUARE_SIZE / largest)

        pixmap = page.get_pixmap(matrix=fitz.Matrix(target_scale, target_scale), alpha=False, clip=clip_rect)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

        coord_scale = self.pdf_render_scale * self.pdf_source_to_work_scale / target_scale
        clip_source_x = (clip_rect.x0 - page_rect.x0) * self.pdf_render_scale
        clip_source_y = (clip_rect.y0 - page_rect.y0) * self.pdf_render_scale
        coord_offset_x = self.pdf_source_to_work_offset_x + clip_source_x * self.pdf_source_to_work_scale
        coord_offset_y = self.pdf_source_to_work_offset_y + clip_source_y * self.pdf_source_to_work_scale
        return self.apply_brush_strokes_to_ocr_image(image, coord_scale, coord_offset_x, coord_offset_y)

    def render_high_quality_raster_png(self) -> Image.Image:
        if self.original_image is None:
            raise OcrError("Изображение не открыто")

        if self.raster_source_image is None:
            box = self.crop_box_pixels()
            image = self.original_image.crop(box).convert("RGB") if box else self.original_image.convert("RGB")
            if max(image.size) < 1600:
                scale = 1600 / max(image.size)
                image = image.resize(
                    (int(round(image.width * scale)), int(round(image.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            return image

        source = self.raster_source_image.convert("RGB")
        box = self.crop_box_pixels()
        if box is None:
            return self.apply_brush_strokes_to_ocr_image(
                source,
                self.source_to_work_scale,
                self.source_to_work_offset_x,
                self.source_to_work_offset_y,
            )

        left, top, right, bottom = box
        source_left, source_top = self.work_to_source_point(left, top)
        source_right, source_bottom = self.work_to_source_point(right, bottom)
        source_box = (
            max(0, int(round(min(source_left, source_right)))),
            max(0, int(round(min(source_top, source_bottom)))),
            min(source.width, int(round(max(source_left, source_right)))),
            min(source.height, int(round(max(source_top, source_bottom)))),
        )
        if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
            return source

        image = source.crop(source_box).convert("RGB")
        coord_offset_x = self.source_to_work_offset_x + source_box[0] * self.source_to_work_scale
        coord_offset_y = self.source_to_work_offset_y + source_box[1] * self.source_to_work_scale
        return self.apply_brush_strokes_to_ocr_image(
            image,
            self.source_to_work_scale,
            coord_offset_x,
            coord_offset_y,
        )

    def build_high_quality_output_image(self) -> Image.Image:
        lasso_crop = self.build_lasso_crop_image() if self.has_valid_lasso() else None
        if lasso_crop is not None:
            return lasso_crop[0]
        if self.source_type == "pdf" and self.pdf_document is not None and self.current_image_is_pdf_page:
            return self.render_high_quality_pdf_png()
        return self.render_high_quality_raster_png()

    def fit_image_to_square(self, image: Image.Image, target_size: int | None = None) -> Image.Image:
        source = image.convert("RGB")
        if target_size is None:
            target_size = max(source.width, source.height, HQ_MIN_SQUARE_SIZE)
            target_size = min(max(target_size, HQ_MIN_SQUARE_SIZE), HQ_MAX_SQUARE_SIZE)
        target_size = max(1, int(target_size))

        scale = min(target_size / max(source.width, 1), target_size / max(source.height, 1))
        new_size = (
            max(1, int(round(source.width * scale))),
            max(1, int(round(source.height * scale))),
        )
        resized = source.resize(new_size, Image.Resampling.LANCZOS)
        result = Image.new("RGB", (target_size, target_size), "white")
        result.paste(
            resized,
            (
                (target_size - resized.width) // 2,
                (target_size - resized.height) // 2,
            ),
        )
        return result

    def quality_option(self, name: str, default: bool = True) -> bool:
        option = getattr(self, name, None)
        if option is None:
            return default
        try:
            return bool(option.get())
        except Exception:
            return default

    def quality_supersample_size(self) -> int:
        try:
            value = int(str(self.quality_supersample_var.get()).strip())
        except Exception:
            value = WORK_IMAGE_SIZE * FINAL_SUPERSAMPLE
        return min(max(value, WORK_IMAGE_SIZE), 4000)

    def auto_trim_quality_source(self, image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
        source = image.convert("RGB")
        if not self.quality_option("quality_auto_trim_var", True):
            return source, (0, 0, source.width, source.height)

        max_probe_side = 1400
        scale = min(1.0, max_probe_side / max(source.width, source.height, 1))
        probe = source
        if scale < 1.0:
            probe = source.resize(
                (
                    max(1, int(round(source.width * scale))),
                    max(1, int(round(source.height * scale))),
                ),
                Image.Resampling.BILINEAR,
            )
        gray = ImageOps.autocontrast(probe.convert("L"), cutoff=1)
        mask = gray.point(lambda value: 255 if value < 246 else 0).filter(ImageFilter.MaxFilter(3))
        bbox = mask.getbbox()
        if bbox is None:
            return source, (0, 0, source.width, source.height)

        inv_scale = 1.0 / max(scale, 0.0001)
        left = int(math.floor(bbox[0] * inv_scale))
        top = int(math.floor(bbox[1] * inv_scale))
        right = int(math.ceil(bbox[2] * inv_scale))
        bottom = int(math.ceil(bbox[3] * inv_scale))
        pad = min(max(int(round(max(source.width, source.height) * 0.025)), 10), 180)
        left = max(0, left - pad)
        top = max(0, top - pad)
        right = min(source.width, right + pad)
        bottom = min(source.height, bottom + pad)
        if right <= left or bottom <= top:
            return source, (0, 0, source.width, source.height)
        if (right - left) >= source.width * 0.985 and (bottom - top) >= source.height * 0.985:
            return source, (0, 0, source.width, source.height)
        return source.crop((left, top, right, bottom)).convert("RGB"), (left, top, right, bottom)

    def remove_tiny_dark_components(
        self,
        image: Image.Image,
        min_area: int | None = None,
        min_span: int | None = None,
        dark_threshold: int = 128,
    ) -> Image.Image:
        gray = image.convert("L")
        max_side = max(gray.size)
        scale = max(1.0, max_side / WORK_IMAGE_SIZE)
        min_area = max(3, int(min_area if min_area is not None else round(scale * scale * 3.0)))
        min_span = max(4, int(min_span if min_span is not None else round(scale * 5.0)))
        dark_threshold = min(max(int(dark_threshold), 1), 254)
        thresholded = gray.point(lambda value: 0 if value < dark_threshold else 255)
        try:
            import cv2
            import numpy as np

            arr = np.asarray(gray)
            dark = (arr < dark_threshold).astype("uint8")
            count, labels, stats, _centroids = cv2.connectedComponentsWithStats(dark, 8)
            if count <= 1:
                return gray

            areas = stats[:, cv2.CC_STAT_AREA]
            widths = stats[:, cv2.CC_STAT_WIDTH]
            heights = stats[:, cv2.CC_STAT_HEIGHT]
            keep = (areas >= min_area) | (widths >= min_span) | (heights >= min_span)
            keep[0] = False
            cleaned = np.where(keep[labels], 0, 255).astype("uint8")
            return Image.fromarray(cleaned, mode="L")
        except Exception:
            return thresholded

    def apply_drawing_mode(self, image: Image.Image) -> Image.Image:
        if not self.quality_option("quality_drawing_mode_var", True):
            return image.convert("RGB")
        gray = ImageOps.grayscale(image)
        gray = ImageOps.autocontrast(gray, cutoff=1)
        gray = ImageEnhance.Contrast(gray).enhance(1.35)
        max_side = max(image.size)
        scale = max(1.0, max_side / WORK_IMAGE_SIZE)
        despeckle_area = max(6, min(90, int(round(scale * scale * 3.0))))
        despeckle_span = max(5, min(26, int(round(scale * 5.0))))
        try:
            import numpy as np

            arr = np.asarray(gray, dtype=np.int16)
            local_radius = max(7, min(23, int(round(max(image.size) / 220))))
            local = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=local_radius)), dtype=np.int16)
            dark = (arr < (local - 10)) | (arr < 176)
            binary_arr = np.where(dark, 0, 255).astype("uint8")
            binary = Image.fromarray(binary_arr, mode="L")
        except Exception:
            binary = gray.point(lambda value: 0 if value < 205 else 255)
        binary = self.remove_tiny_dark_components(
            binary,
            min_area=despeckle_area,
            min_span=despeckle_span,
        )
        return Image.merge("RGB", (binary, binary, binary))

    def sharpen_lines_only(self, image: Image.Image) -> Image.Image:
        if not self.quality_option("quality_line_sharpen_var", True):
            return image.convert("RGB")
        source = image.convert("RGB")
        gray = source.convert("L")
        line_binary = gray.point(lambda value: 0 if value < 214 else 255)
        line_binary = self.remove_tiny_dark_components(line_binary, min_area=5, min_span=5)
        mask = ImageOps.invert(line_binary)
        mask = mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(radius=0.55))
        sharpened = source.filter(ImageFilter.UnsharpMask(radius=0.45, percent=320, threshold=1))
        return Image.composite(sharpened, source, mask)

    def font_for_redrawn_number(self, size: int) -> ImageFont.ImageFont:
        for font_name in ("arialbd.ttf", "calibrib.ttf", "tahoma.ttf", "arial.ttf"):
            try:
                return ImageFont.truetype(font_name, size)
            except Exception:
                continue
        return ImageFont.load_default()

    def redraw_numbers_on_final_image(self, image: Image.Image, spots: list[Hotspot]) -> Image.Image:
        if not self.quality_option("quality_redraw_numbers_var", False) or not spots:
            return image
        result = image.convert("RGB")
        draw = ImageDraw.Draw(result)
        for spot in spots:
            if not spot.number:
                continue
            if spot.x < 0 or spot.y < 0 or spot.x > result.width or spot.y > result.height:
                continue
            font_size = int(round(max(12.0, min(28.0, max(spot.width, spot.height) * 1.05 + 4.0))))
            font = self.font_for_redrawn_number(font_size)
            bbox = draw.textbbox((0, 0), spot.number, font=font, stroke_width=1)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = spot.x - text_width / 2
            y = spot.y - text_height / 2 - bbox[1]
            draw.text(
                (x, y),
                spot.number,
                fill="black",
                font=font,
                stroke_width=2,
                stroke_fill="white",
            )
        return result

    def finalize_quality_800_image(
        self,
        image: Image.Image,
        source_spots: list[Hotspot] | None = None,
    ) -> tuple[Image.Image, list[Hotspot], Image.Image, float, float, float]:
        prepared_source, trim_box = self.auto_trim_quality_source(image)
        trim_left, trim_top, _trim_right, _trim_bottom = trim_box
        adjusted_spots: list[Hotspot] = []
        for source_spot in source_spots or []:
            spot = Hotspot(**asdict(source_spot))
            spot.x -= trim_left
            spot.y -= trim_top
            if 0 <= spot.x <= prepared_source.width and 0 <= spot.y <= prepared_source.height:
                adjusted_spots.append(spot)

        drawing_enabled = self.quality_option("quality_drawing_mode_var", False)
        line_sharpen_enabled = self.quality_option("quality_line_sharpen_var", False)
        if not drawing_enabled and not line_sharpen_enabled:
            result, _scale, _offset_x, _offset_y = render_square_from_source_tiles(
                prepared_source,
                size=WORK_IMAGE_SIZE,
                tile_size=FINAL_TILE_SIZE,
                tile_overlap=12,
                sharpen=False,
            )
        else:
            supersampled_size = self.quality_supersample_size()
            supersampled, _scale, _offset_x, _offset_y = render_square_from_source_tiles(
                prepared_source,
                size=supersampled_size,
                tile_size=FINAL_TILE_SIZE,
                tile_overlap=12,
                sharpen=True,
                sharpen_radius=0.55,
                sharpen_percent=120,
                sharpen_threshold=1,
            )
            supersampled = self.apply_drawing_mode(supersampled)
            result = supersampled.resize((WORK_IMAGE_SIZE, WORK_IMAGE_SIZE), Image.Resampling.LANCZOS)
            if drawing_enabled:
                result = self.remove_tiny_dark_components(
                    result,
                    min_area=4,
                    min_span=5,
                    dark_threshold=170,
                ).convert("RGB")
        final_scale = min(
            WORK_IMAGE_SIZE / max(prepared_source.width, 1),
            WORK_IMAGE_SIZE / max(prepared_source.height, 1),
        )
        final_offset_x = (WORK_IMAGE_SIZE - prepared_source.width * final_scale) / 2
        final_offset_y = (WORK_IMAGE_SIZE - prepared_source.height * final_scale) / 2
        final_spots: list[Hotspot] = []
        for adjusted_spot in adjusted_spots:
            spot = Hotspot(**asdict(adjusted_spot))
            spot.x = spot.x * final_scale + final_offset_x
            spot.y = spot.y * final_scale + final_offset_y
            spot.width = max(1.0, spot.width * final_scale)
            spot.height = max(1.0, spot.height * final_scale)
            final_spots.append(spot)
        result = self.sharpen_lines_only(result)
        result = self.redraw_numbers_on_final_image(result, final_spots)
        return result, final_spots, prepared_source, final_scale, final_offset_x, final_offset_y

    def fit_image_to_final_800_square(self, image: Image.Image) -> Image.Image:
        result, _spots, _source, _scale, _offset_x, _offset_y = self.finalize_quality_800_image(image)
        return result

    def save_high_quality_png(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("Нет изображения", "Сначала откройте изображение или PDF.")
            return

        file_name = filedialog.asksaveasfilename(
            title="Сохранить PNG в хорошем качестве",
            defaultextension=".png",
            filetypes=(("PNG", "*.png"), ("All files", "*.*")),
        )
        if not file_name:
            return

        try:
            image = self.build_high_quality_output_image()
            image = self.fit_image_to_final_800_square(image)
            image.save(file_name, format="PNG", compress_level=1, dpi=(300, 300))
        except Exception as exc:
            messagebox.showerror("Ошибка PNG", f"Не удалось сохранить PNG:\n{exc}")
            return

        self.status.set(f"PNG HD сохранён в формате 800x800 из high-res источника: {file_name}")

    def add_spot_at(self, x: float, y: float) -> None:
        if self.original_image is None:
            return
        number = self.number_entry.get().strip()
        if not number:
            existing_numbers = [int(spot.number) for spot in self.spots if spot.number.isdigit()]
            number = str(max(existing_numbers, default=0) + 1)

        self.articles = parse_articles(self.articles_text.get("1.0", tk.END))
        article = self.article_entry.get().strip() or get_article_for_number(self.articles, number)
        self.spots.append(Hotspot(number=number, x=x, y=y, width=32, height=32, article=article, source="manual"))
        self.select_spot(len(self.spots) - 1)
        self.refresh_tree()
        self.redraw_canvas()
        self.status.set(f"Добавлена цифра №{number}.")

    def select_spot(self, index: int, center: bool = False) -> None:
        if not (0 <= index < len(self.spots)):
            return
        self.selected_index = index
        spot = self.spots[index]
        self.number_entry.delete(0, tk.END)
        self.number_entry.insert(0, spot.number)
        self.article_entry.delete(0, tk.END)
        self.article_entry.insert(0, spot.article)
        self.load_candidate_options(spot.number)
        iid = str(index)
        if self.tree.exists(iid):
            self.tree.selection_set(iid)
            self.tree.see(iid)
        if center:
            self.center_view_on_point(spot.x, spot.y)
        self.redraw_canvas()

    def on_tree_select(self, _event: tk.Event) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        try:
            index = int(selected[0])
        except ValueError:
            return
        if index != self.selected_index:
            self.select_spot(index)

    def load_candidate_options(self, number: str) -> None:
        normalized = normalize_number(number)
        options = self.candidate_spots_by_number.get(normalized, [])
        self.current_candidate_options = options
        values: list[str] = []
        if self.original_image is not None:
            for index, candidate in enumerate(options, start=1):
                x_percent, y_percent = candidate.center_percent(
                    self.original_image.width,
                    self.original_image.height,
                )
                source = candidate.source.split(";", 1)[0]
                values.append(f"{index}. X {x_percent:.2f}% Y {y_percent:.2f}% - {source}")
        self.candidate_combo["values"] = values
        self.candidate_var.set(values[0] if values else "")

    def apply_selected_candidate(self) -> None:
        if self.selected_index is None or not (0 <= self.selected_index < len(self.spots)):
            messagebox.showinfo("Нет цифры", "Выберите цифру в списке или на картинке.")
            return
        if not self.current_candidate_options:
            self.status.set("Для выбранной цифры нет альтернативных кандидатов.")
            return

        selected_value = self.candidate_var.get()
        candidate_index = 0
        if selected_value:
            try:
                candidate_index = max(0, int(selected_value.split(".", 1)[0]) - 1)
            except ValueError:
                candidate_index = 0
        if not (0 <= candidate_index < len(self.current_candidate_options)):
            return

        candidate = self.current_candidate_options[candidate_index]
        spot = self.spots[self.selected_index]
        spot.x = candidate.x
        spot.y = candidate.y
        spot.width = candidate.width
        spot.height = candidate.height
        spot.source = f"{candidate.source}; selected candidate"
        self.refresh_tree()
        self.select_spot(self.selected_index, center=True)
        self.status.set(f"Цифра №{spot.number} перенесена на выбранный кандидат.")

    def apply_editor(self) -> None:
        if self.selected_index is None or not (0 <= self.selected_index < len(self.spots)):
            messagebox.showinfo("Нет цифры", "Выберите цифру в списке или на картинке.")
            return

        number = self.number_entry.get().strip()
        if not number:
            messagebox.showwarning("Номер пустой", "Укажите номер цифры.")
            return

        spot = self.spots[self.selected_index]
        spot.number = number
        spot.article = self.article_entry.get().strip()
        self.articles[normalize_number(number)] = spot.article
        self.refresh_tree()
        self.redraw_canvas()
        self.status.set(f"Цифра №{number} сохранена.")

    def delete_selected(self) -> None:
        if self.selected_index is None or not (0 <= self.selected_index < len(self.spots)):
            return
        removed = self.spots.pop(self.selected_index)
        self.selected_index = None
        self.refresh_tree()
        self.redraw_canvas()
        self.status.set(f"Удалена цифра №{removed.number}.")

    def export_json(self) -> None:
        if not self._can_export():
            return
        assert self.original_image is not None

        file_name = filedialog.asksaveasfilename(
            title="Сохранить JSON",
            defaultextension=".json",
            filetypes=(("JSON", "*.json"), ("All files", "*.*")),
        )
        if not file_name:
            return

        payload = {
            "image": str(self.image_path) if self.image_path else "",
            "source": str(self.source_path) if self.source_path else "",
            "source_type": self.source_type,
            "pdf_page": self.pdf_page_index + 1 if self.source_type == "pdf" else None,
            "image_width": self.original_image.width,
            "image_height": self.original_image.height,
            "spots": [asdict(spot) for spot in self.spots],
        }
        Path(file_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.status.set(f"JSON сохранён: {file_name}")

    def export_csv(self) -> None:
        if not self._can_export():
            return
        assert self.original_image is not None

        file_name = filedialog.asksaveasfilename(
            title="Сохранить CSV",
            defaultextension=".csv",
            filetypes=(("CSV", "*.csv"), ("All files", "*.*")),
        )
        if not file_name:
            return

        fieldnames = [
            "number",
            "article",
            "source",
            "source_type",
            "pdf_page",
            "x_percent",
            "y_percent",
            "x_px",
            "y_px",
            "width_px",
            "height_px",
            "ocr_source",
        ]
        with Path(file_name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            for row in self.build_export_rows():
                writer.writerow(row)
        self.status.set(f"CSV сохранён: {file_name}")

    def _can_export(self) -> bool:
        if self.original_image is None or self.image_path is None:
            messagebox.showinfo("Нет изображения", "Сначала откройте PDF или картинку.")
            return False
        if not self.spots:
            messagebox.showinfo("Нет цифр", "Сначала распознайте или добавьте цифры.")
            return False
        return True

    def build_export_rows(self) -> list[dict[str, str | int | float | None]]:
        assert self.original_image is not None
        rows: list[dict[str, str | int | float | None]] = []
        page_number = self.pdf_page_index + 1 if self.source_type == "pdf" else None
        source = str(self.source_path) if self.source_path else str(self.image_path or "")
        for spot in self.spots:
            x_percent, y_percent = spot.center_percent(self.original_image.width, self.original_image.height)
            rows.append(
                {
                    "number": spot.number,
                    "article": spot.article,
                    "source": source,
                    "source_type": self.source_type,
                    "pdf_page": page_number,
                    "x_percent": round(x_percent, 4),
                    "y_percent": round(y_percent, 4),
                    "x_px": round(spot.x, 2),
                    "y_px": round(spot.y, 2),
                    "width_px": round(spot.width, 2),
                    "height_px": round(spot.height, 2),
                    "ocr_source": spot.source,
                }
            )
        return rows


def main() -> None:
    app = PartsHotspotApp()
    app.mainloop()


if __name__ == "__main__":
    main()
