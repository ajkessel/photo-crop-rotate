#!/usr/bin/env python3
"""Crop scanned-photo borders and orient images using face detection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"
)


@dataclass(frozen=True)
class InputImage:
    path: Path
    root: Path | None


@dataclass(frozen=True)
class CropResult:
    image: np.ndarray
    box: tuple[int, int, int, int]
    status: str


@dataclass(frozen=True)
class SplitRegion:
    image: np.ndarray
    box: tuple[int, int, int, int]
    index: int
    total: int
    status: str


@dataclass(frozen=True)
class RotationResult:
    image: np.ndarray
    angle: int
    status: str
    faces: int
    score: float


@dataclass(frozen=True)
class ProcessResult:
    input_path: Path
    output_path: Path
    split_status: str
    split_index: int
    split_total: int
    split_box: tuple[int, int, int, int]
    crop_status: str
    rotation_status: str
    crop_box: tuple[int, int, int, int]
    angle: int
    faces: int
    score: float
    written: bool
    error: str = ""


class FaceOrienter:
    def __init__(
        self,
        model_path: Path,
        min_confidence: float,
        contrast_fallback: bool = True,
        haar_fallback: bool = True,
    ) -> None:
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.contrast_fallback = contrast_fallback
        self.haar_fallback = haar_fallback
        self._detector = None
        self._mp = None

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()

    def orient(self, image: np.ndarray) -> RotationResult:
        detector, mp = self._load_detector()
        primary = best_mediapipe_orientation(detector, mp, image, image, "rotated", "upright")
        if primary.faces > 0:
            return primary

        if self.contrast_fallback:
            contrast_image = contrast_stretch_image(image)
            contrast = best_mediapipe_orientation(
                detector,
                mp,
                contrast_image,
                image,
                "contrast_rotated",
                "contrast_upright",
            )
            if contrast.faces > 0:
                return contrast

        if self.haar_fallback:
            fallback = orient_with_haar_cascade(image)
            if fallback.faces > 0:
                return fallback
        return RotationResult(image=image, angle=0, status="no_face_detected", faces=0, score=0.0)

    def _load_detector(self):
        if self._detector is not None:
            return self._detector, self._mp

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=self.min_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        self._mp = mp
        return self._detector, self._mp


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop scanner borders and rotate scanned photos so detected faces are upright."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Image files or directories containing images.")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Directory for processed images.")
    parser.add_argument("--recursive", action="store_true", help="Scan input directories recursively.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    parser.add_argument(
        "--format",
        choices=("same", "jpeg", "png"),
        default="same",
        help="Output image format. Default: same as input.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality from 1 to 100.")
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.015,
        help="Padding around detected photo content, as a fraction of the larger image dimension.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum MediaPipe face detection confidence.",
    )
    parser.add_argument("--model", type=Path, help="Path to a MediaPipe face detector model file.")
    parser.add_argument(
        "--model-url",
        default=DEFAULT_MODEL_URL,
        help="URL used to download the default face detector model when --model is not supplied.",
    )
    parser.add_argument("--debug-dir", type=Path, help="Write crop masks and intermediate thumbnails.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without writing output images.")
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Disable automatic splitting of scans that contain multiple separated photos.",
    )
    parser.add_argument(
        "--skip-orientation",
        action="store_true",
        help="Crop images but do not run MediaPipe face orientation.",
    )
    parser.add_argument(
        "--no-haar-fallback",
        action="store_true",
        help="Disable the OpenCV Haar fallback used when MediaPipe detects no faces.",
    )
    parser.add_argument(
        "--no-contrast-fallback",
        action="store_true",
        help="Disable contrast-enhanced MediaPipe retry used for dark images.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional CSV report path. Defaults to OUTPUT_DIR/process_report.csv when writing images.",
    )
    return parser.parse_args(argv)


def expand_inputs(inputs: Sequence[Path], recursive: bool) -> list[InputImage]:
    expanded: list[InputImage] = []
    seen: set[Path] = set()
    for input_path in inputs:
        path = input_path.expanduser()
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            candidates = sorted(p for p in path.glob(pattern) if p.is_file() and is_supported_image(p))
            for candidate in candidates:
                resolved = candidate.resolve()
                if resolved not in seen:
                    expanded.append(InputImage(path=candidate, root=path))
                    seen.add(resolved)
        elif path.is_file() and is_supported_image(path):
            resolved = path.resolve()
            if resolved not in seen:
                expanded.append(InputImage(path=path, root=None))
                seen.add(resolved)
        elif path.is_file():
            print(f"Skipping unsupported file: {path}", file=sys.stderr)
        else:
            raise FileNotFoundError(f"Input does not exist: {path}")
    return expanded


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def output_path_for(input_image: InputImage, output_dir: Path, output_format: str) -> Path:
    if input_image.root is None:
        relative = Path(input_image.path.name)
    else:
        relative = input_image.path.relative_to(input_image.root)

    suffix = relative.suffix
    if output_format == "jpeg":
        suffix = ".jpg"
    elif output_format == "png":
        suffix = ".png"

    return output_dir / relative.with_suffix(suffix)


def split_output_path(path: Path, index: int, total: int) -> Path:
    if total <= 1:
        return path
    return path.with_name(f"{path.stem}_{index}{path.suffix}")


def read_image(path: Path) -> np.ndarray:
    with Image.open(path) as pil_image:
        pil_image = ImageOps.exif_transpose(pil_image)
        rgb = pil_image.convert("RGB")
        array = np.array(rgb)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def write_image(path: Path, image: np.ndarray, output_format: str, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    params: list[int] = []
    if output_format == "jpeg" or suffix in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, jpeg_quality))]
    if not cv2.imwrite(str(path), image, params):
        raise OSError(f"Failed to write image: {path}")


def crop_scanner_border(image: np.ndarray, padding_fraction: float) -> CropResult:
    height, width = image.shape[:2]
    if height < 8 or width < 8:
        return CropResult(image=image, box=(0, 0, width, height), status="too_small")

    mask = build_content_mask(image)
    box = content_box_from_mask(mask, width, height)
    if box is None:
        return CropResult(image=image, box=(0, 0, width, height), status="crop_failed")

    x, y, w, h = padded_box(box, width, height, padding_fraction)
    if w >= width * 0.985 and h >= height * 0.985:
        return CropResult(image=image, box=(0, 0, width, height), status="no_border_detected")

    return CropResult(image=image[y : y + h, x : x + w], box=(x, y, w, h), status="cropped")


def build_content_mask(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    border = collect_border_pixels(image)
    background_bgr = np.median(border.reshape(-1, 3), axis=0)
    background_lab = cv2.cvtColor(np.uint8([[background_bgr]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    distance = np.linalg.norm(lab - background_lab, axis=2)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
    border_gray = cv2.cvtColor(border, cv2.COLOR_BGR2GRAY)
    border_std = float(np.std(border_gray))

    distance_threshold = max(10.0, border_std * 2.5 + 7.0)
    non_background = distance > distance_threshold
    darker_than_border = gray < max(245, float(np.median(border_gray)) - 4.0)
    saturated = saturation > 25
    mask = (non_background | saturated | darker_than_border).astype(np.uint8) * 255

    kernel_size = max(3, int(round(min(height, width) * 0.006)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def split_scan_regions(image: np.ndarray, padding_fraction: float) -> list[SplitRegion]:
    height, width = image.shape[:2]
    mask = build_split_mask(image)
    boxes = split_boxes_from_mask(mask, width, height)
    if len(boxes) <= 1:
        return [SplitRegion(image=image, box=(0, 0, width, height), index=1, total=1, status="not_split")]

    ordered_boxes = sorted(boxes, key=lambda box: (box[1], box[0]))
    total = len(ordered_boxes)
    regions: list[SplitRegion] = []
    for index, box in enumerate(ordered_boxes, start=1):
        x, y, w, h = padded_box(box, width, height, padding_fraction)
        regions.append(
            SplitRegion(
                image=image[y : y + h, x : x + w],
                box=(x, y, w, h),
                index=index,
                total=total,
                status="split",
            )
        )
    return regions


def build_split_mask(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    mask = build_content_mask(image)
    close_size = max(9, int(round(min(height, width) * 0.018)))
    if close_size % 2 == 0:
        close_size += 1
    kernel = np.ones((close_size, close_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def split_boxes_from_mask(mask: np.ndarray, width: int, height: int) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    image_area = width * height
    min_area = max(256.0, image_area * 0.025)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.12 or h < height * 0.12:
            continue
        if w >= width * 0.95 and h >= height * 0.95:
            continue
        boxes.append((x, y, w, h))
    return boxes


def collect_border_pixels(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    edge = max(4, int(round(min(height, width) * 0.04)))
    top = image[:edge, :, :]
    bottom = image[height - edge :, :, :]
    left = image[:, :edge, :]
    right = image[:, width - edge :, :]
    return np.concatenate(
        [
            top.reshape(-1, 1, 3),
            bottom.reshape(-1, 1, 3),
            left.reshape(-1, 1, 3),
            right.reshape(-1, 1, 3),
        ],
        axis=0,
    )


def content_box_from_mask(mask: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = max(64.0, width * height * 0.01)
    useful_boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.08 or h < height * 0.08:
            continue
        useful_boxes.append((x, y, w, h))

    if not useful_boxes:
        return None

    x1 = min(x for x, _, _, _ in useful_boxes)
    y1 = min(y for _, y, _, _ in useful_boxes)
    x2 = max(x + w for x, _, w, _ in useful_boxes)
    y2 = max(y + h for _, y, _, h in useful_boxes)
    return x1, y1, x2 - x1, y2 - y1


def padded_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    padding_fraction: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    padding = max(0, int(round(max(width, height) * padding_fraction)))
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)
    return x1, y1, x2 - x1, y2 - y1


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation angle: {angle}")


def best_mediapipe_orientation(
    detector,
    mp,
    detection_image: np.ndarray,
    output_image: np.ndarray,
    rotated_status: str,
    upright_status: str,
) -> RotationResult:
    best_image = output_image
    best_angle = 0
    best_score = -math.inf
    best_faces = 0

    for angle in (0, 90, 180, 270):
        detection_candidate = rotate_image(detection_image, angle)
        output_candidate = rotate_image(output_image, angle)
        rgb = cv2.cvtColor(detection_candidate, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        score = score_detection_result(result, detection_candidate.shape)
        faces = len(result.detections) if result.detections else 0
        if score > best_score:
            best_image = output_candidate
            best_angle = angle
            best_score = score
            best_faces = faces

    if best_faces == 0:
        return RotationResult(image=output_image, angle=0, status="no_face_detected", faces=0, score=0.0)
    return RotationResult(
        image=best_image,
        angle=best_angle,
        status=rotated_status if best_angle else upright_status,
        faces=best_faces,
        score=float(best_score),
    )


def contrast_stretch_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low, high = np.percentile(gray, [1, 99])
    if high - low < 8:
        return image.copy()
    stretched = np.clip((gray.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)


def score_detection_result(result, shape: tuple[int, int, int]) -> float:
    detections = result.detections or []
    if not detections:
        return -1.0

    height, width = shape[:2]
    image_area = float(width * height)
    score = 0.0
    for detection in detections:
        confidence = float(detection.categories[0].score) if detection.categories else 0.0
        bbox = detection.bounding_box
        face_area = max(0.0, float(bbox.width * bbox.height)) / image_area
        score += confidence * 100.0
        score += min(face_area * 500.0, 50.0)
        score += landmark_orientation_bonus(detection.keypoints, width, height)
    return score + len(detections) * 20.0


def landmark_orientation_bonus(keypoints, width: int, height: int) -> float:
    if not keypoints or len(keypoints) < 4:
        return 0.0

    points = [(float(k.x) * width, float(k.y) * height) for k in keypoints]
    left_eye, right_eye, nose = points[0], points[1], points[2]
    eye_dx = abs(right_eye[0] - left_eye[0])
    eye_dy = abs(right_eye[1] - left_eye[1])
    if eye_dx < 1:
        return -15.0

    bonus = 0.0
    if eye_dy / eye_dx < 0.35:
        bonus += 25.0
    if nose[1] > min(left_eye[1], right_eye[1]):
        bonus += 35.0
    else:
        bonus -= 40.0
    if len(points) >= 6:
        mouth_left, mouth_right = points[4], points[5]
        mouth_y = (mouth_left[1] + mouth_right[1]) / 2.0
        if mouth_y > nose[1]:
            bonus += 25.0
        else:
            bonus -= 35.0
    return bonus


def orient_with_haar_cascade(image: np.ndarray) -> RotationResult:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        return RotationResult(image=image, angle=0, status="no_face_detected", faces=0, score=0.0)

    best_image = image
    best_angle = 0
    best_score = -math.inf
    best_faces = 0
    for angle in (0, 90, 180, 270):
        candidate = rotate_image(image, angle)
        score, faces = score_haar_orientation(candidate, cascade)
        if score > best_score:
            best_image = candidate
            best_angle = angle
            best_score = score
            best_faces = faces

    if best_faces == 0:
        return RotationResult(image=image, angle=0, status="no_face_detected", faces=0, score=0.0)
    return RotationResult(
        image=best_image,
        angle=best_angle,
        status="haar_rotated" if best_angle else "haar_upright",
        faces=best_faces,
        score=float(best_score),
    )


def score_haar_orientation(image: np.ndarray, cascade: cv2.CascadeClassifier) -> tuple[float, int]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    preparations = (
        gray,
        cv2.equalizeHist(gray),
        cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray),
    )

    total_score = 0.0
    total_faces = 0
    for prepared in preparations:
        scaled, inverse_scale = scale_for_haar(prepared)
        boxes = cascade.detectMultiScale(
            scaled,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(20, 20),
        )
        total_faces += len(boxes)
        total_score += score_haar_boxes(boxes, inverse_scale, image.shape)
    return total_score, total_faces


def scale_for_haar(gray: np.ndarray, max_dimension: int = 1600) -> tuple[np.ndarray, float]:
    height, width = gray.shape[:2]
    current_max = max(height, width)
    if current_max <= max_dimension:
        return gray, 1.0

    scale = max_dimension / current_max
    scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return scaled, 1.0 / scale


def score_haar_boxes(
    boxes: Iterable[tuple[int, int, int, int]],
    inverse_scale: float,
    shape: tuple[int, int, int],
) -> float:
    height, width = shape[:2]
    image_area = float(width * height)
    score = 0.0
    for x, y, w, h in boxes:
        face_area = (w * inverse_scale) * (h * inverse_scale)
        area_fraction = face_area / image_area
        if area_fraction < 0.0004 or area_fraction > 0.18:
            continue
        aspect = w / h if h else 0.0
        if aspect < 0.65 or aspect > 1.45:
            continue
        score += 10.0 + min(area_fraction * 900.0, 45.0)
    return score


def ensure_model(model_path: Path | None, model_url: str) -> Path:
    if model_path is not None:
        if not model_path.exists():
            raise FileNotFoundError(f"MediaPipe model not found: {model_path}")
        return model_path

    bundled_model = bundled_resource_path(Path("models") / Path(model_url).name)
    if bundled_model.exists() and bundled_model.stat().st_size > 0:
        return bundled_model

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    destination = models_dir / Path(model_url).name
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    print(f"Downloading MediaPipe face detector model to {destination}", file=sys.stderr)
    with urllib.request.urlopen(model_url) as response, destination.open("wb") as file:
        file.write(response.read())
    return destination


def bundled_resource_path(relative_path: Path) -> Path:
    base_path = Path(getattr(sys, "_MEIPASS", Path.cwd()))
    return base_path / relative_path


def debug_stem(input_path: Path) -> str:
    digest = hashlib.sha1(str(input_path).encode("utf-8")).hexdigest()[:10]
    return f"{input_path.stem}-{digest}"


def write_debug_images(
    debug_dir: Path,
    input_path: Path,
    original: np.ndarray,
    crop: CropResult,
    suffix: str = "",
) -> None:
    stem = debug_stem(input_path)
    if suffix:
        stem = f"{stem}-{suffix}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    x, y, w, h = crop.box
    annotated = original.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), max(2, min(original.shape[:2]) // 250))
    cv2.imwrite(str(debug_dir / f"{stem}-crop-box.jpg"), annotated)
    cv2.imwrite(str(debug_dir / f"{stem}-cropped.jpg"), crop.image)


def process_one(
    input_image: InputImage,
    output_path: Path,
    orienter: FaceOrienter | None,
    args: argparse.Namespace,
) -> list[ProcessResult]:
    image = read_image(input_image.path)
    regions = (
        [SplitRegion(image=image, box=(0, 0, image.shape[1], image.shape[0]), index=1, total=1, status="split_disabled")]
        if args.no_split
        else split_scan_regions(image, args.crop_padding)
    )
    output_paths = [split_output_path(output_path, region.index, region.total) for region in regions]
    if not args.dry_run and not args.overwrite:
        for region_output_path in output_paths:
            if region_output_path.exists():
                raise FileExistsError(f"Output already exists: {region_output_path}")

    results: list[ProcessResult] = []
    for region, region_output_path in zip(regions, output_paths, strict=True):
        crop = crop_scanner_border(region.image, args.crop_padding)

        if args.debug_dir:
            suffix = f"part-{region.index}" if region.total > 1 else ""
            write_debug_images(args.debug_dir, input_image.path, region.image, crop, suffix=suffix)

        if orienter is None:
            rotation = RotationResult(crop.image, angle=0, status="orientation_skipped", faces=0, score=0.0)
        else:
            rotation = orienter.orient(crop.image)

        if args.dry_run:
            written = False
        else:
            write_image(region_output_path, rotation.image, args.format, args.jpeg_quality)
            written = True

        results.append(
            ProcessResult(
                input_path=input_image.path,
                output_path=region_output_path,
                split_status=region.status,
                split_index=region.index,
                split_total=region.total,
                split_box=region.box,
                crop_status=crop.status,
                rotation_status=rotation.status,
                crop_box=crop.box,
                angle=rotation.angle,
                faces=rotation.faces,
                score=rotation.score,
                written=written,
            )
        )
    return results


def write_report(path: Path, results: Sequence[ProcessResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "input",
                "output",
                "split_status",
                "split_index",
                "split_total",
                "split_box",
                "crop_status",
                "rotation_status",
                "crop_box",
                "angle",
                "faces",
                "score",
                "written",
                "error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "input": str(result.input_path),
                    "output": str(result.output_path),
                    "split_status": result.split_status,
                    "split_index": result.split_index,
                    "split_total": result.split_total,
                    "split_box": " ".join(str(value) for value in result.split_box),
                    "crop_status": result.crop_status,
                    "rotation_status": result.rotation_status,
                    "crop_box": " ".join(str(value) for value in result.crop_box),
                    "angle": result.angle,
                    "faces": result.faces,
                    "score": f"{result.score:.3f}",
                    "written": result.written,
                    "error": result.error,
                }
            )


def print_summary(results: Sequence[ProcessResult]) -> None:
    failures = [result for result in results if result.error]
    cropped = sum(1 for result in results if result.crop_status == "cropped")
    split_outputs = sum(1 for result in results if result.split_total > 1)
    rotated = sum(
        1
        for result in results
        if result.rotation_status in {"rotated", "contrast_rotated", "haar_rotated"}
    )
    no_faces = sum(1 for result in results if result.rotation_status == "no_face_detected")
    print(
        f"Processed {len(results)} output image(s): {split_outputs} split, {cropped} cropped, {rotated} rotated, "
        f"{no_faces} without detected faces, {len(failures)} failed."
    )
    for failure in failures:
        print(f"FAILED {failure.input_path}: {failure.error}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not 0.0 <= args.crop_padding <= 0.25:
        raise ValueError("--crop-padding must be between 0 and 0.25")
    if not 0.0 < args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must be between 0 and 1")

    inputs = expand_inputs(args.inputs, args.recursive)
    if not inputs:
        print("No supported input images found.", file=sys.stderr)
        return 2

    model_path = None if args.skip_orientation else ensure_model(args.model, args.model_url)
    orienter = (
        None
        if args.skip_orientation
        else FaceOrienter(
            model_path,
            args.min_confidence,
            contrast_fallback=not args.no_contrast_fallback,
            haar_fallback=not args.no_haar_fallback,
        )
    )
    report_path = args.report or (args.output_dir / "process_report.csv")
    results: list[ProcessResult] = []

    try:
        for input_image in tqdm(inputs, desc="Processing", unit="image"):
            output_path = output_path_for(input_image, args.output_dir, args.format)
            try:
                results.extend(process_one(input_image, output_path, orienter, args))
            except Exception as exc:
                results.append(
                    ProcessResult(
                        input_path=input_image.path,
                        output_path=output_path,
                        split_status="failed",
                        split_index=1,
                        split_total=1,
                        split_box=(0, 0, 0, 0),
                        crop_status="failed",
                        rotation_status="failed",
                        crop_box=(0, 0, 0, 0),
                        angle=0,
                        faces=0,
                        score=0.0,
                        written=False,
                        error=str(exc),
                    )
                )
    finally:
        if orienter is not None:
            orienter.close()

    if not args.dry_run:
        write_report(report_path, results)
    else:
        for result in results:
            print(
                f"{result.input_path} -> {result.output_path} "
                f"split={result.split_status}:{result.split_index}/{result.split_total} "
                f"crop={result.crop_status} rotate={result.rotation_status} angle={result.angle}"
            )

    print_summary(results)
    return 1 if any(result.error for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
