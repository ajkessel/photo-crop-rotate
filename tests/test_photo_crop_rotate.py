from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from photo_crop_rotate import (
    InputImage,
    contrast_stretch_image,
    crop_scanner_border,
    expand_inputs,
    main,
    output_path_for,
    rotate_image,
    score_haar_boxes,
    split_output_path,
    split_scan_regions,
)


def test_crop_scanner_border_finds_photo_content() -> None:
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    image[24:96, 36:126] = (50, 120, 180)

    result = crop_scanner_border(image, padding_fraction=0.0)

    assert result.status == "cropped"
    x, y, w, h = result.box
    assert 30 <= x <= 40
    assert 20 <= y <= 28
    assert 84 <= w <= 96
    assert 66 <= h <= 78
    assert result.image.shape[:2] == (h, w)


def test_crop_scanner_border_handles_grayscale_content() -> None:
    image = np.full((100, 140, 3), 248, dtype=np.uint8)
    image[15:88, 18:120] = 190
    cv2.line(image, (30, 30), (110, 72), (40, 40, 40), 3)

    result = crop_scanner_border(image, padding_fraction=0.0)

    assert result.status == "cropped"
    x, y, w, h = result.box
    assert x <= 22
    assert y <= 19
    assert x + w >= 116
    assert y + h >= 84


def test_crop_scanner_border_falls_back_when_content_is_not_found() -> None:
    image = np.full((80, 90, 3), 255, dtype=np.uint8)

    result = crop_scanner_border(image, padding_fraction=0.0)

    assert result.status == "crop_failed"
    assert result.box == (0, 0, 90, 80)
    assert result.image.shape == image.shape


def test_expand_inputs_supports_files_directories_and_recursion(tmp_path: Path) -> None:
    root = tmp_path / "scans"
    nested = root / "nested"
    nested.mkdir(parents=True)
    make_image(root / "a.jpg")
    make_image(nested / "b.png")
    (root / "notes.txt").write_text("ignore me", encoding="utf-8")

    non_recursive = expand_inputs([root], recursive=False)
    recursive = expand_inputs([root], recursive=True)

    assert [item.path.name for item in non_recursive] == ["a.jpg"]
    assert [item.path.name for item in recursive] == ["a.jpg", "b.png"]
    assert all(item.root == root for item in recursive)


def test_output_path_preserves_directory_relative_path(tmp_path: Path) -> None:
    root = tmp_path / "scans"
    path = root / "family" / "photo.tif"
    item = InputImage(path=path, root=root)

    assert output_path_for(item, tmp_path / "out", "same") == tmp_path / "out" / "family" / "photo.tif"
    assert output_path_for(item, tmp_path / "out", "jpeg") == tmp_path / "out" / "family" / "photo.jpg"


def test_split_output_path_appends_index_only_for_multiple_outputs() -> None:
    path = Path("out/input.jpg")

    assert split_output_path(path, 1, 1) == path
    assert split_output_path(path, 1, 2) == Path("out/input_1.jpg")
    assert split_output_path(path, 2, 2) == Path("out/input_2.jpg")


def test_split_scan_regions_finds_separated_photos() -> None:
    image = np.full((180, 260, 3), 255, dtype=np.uint8)
    image[25:155, 20:105] = (45, 100, 150)
    image[25:155, 150:240] = (150, 90, 45)

    regions = split_scan_regions(image, padding_fraction=0.0)

    assert len(regions) == 2
    assert [region.index for region in regions] == [1, 2]
    assert all(region.total == 2 for region in regions)
    assert all(region.status == "split" for region in regions)
    assert regions[0].box[0] < regions[1].box[0]


def test_split_scan_regions_keeps_single_photo_as_one_region() -> None:
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    image[20:100, 30:130] = (45, 100, 150)

    regions = split_scan_regions(image, padding_fraction=0.0)

    assert len(regions) == 1
    assert regions[0].status == "not_split"
    assert regions[0].box == (0, 0, 160, 120)


@pytest.mark.parametrize(
    ("angle", "shape"),
    [
        (0, (10, 20, 3)),
        (90, (20, 10, 3)),
        (180, (10, 20, 3)),
        (270, (20, 10, 3)),
    ],
)
def test_rotate_image_right_angles(angle: int, shape: tuple[int, int, int]) -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    rotated = rotate_image(image, angle)

    assert rotated.shape == shape


def test_score_haar_boxes_rewards_plausible_faces() -> None:
    score = score_haar_boxes([(10, 10, 40, 42), (100, 20, 50, 45)], 1.0, (300, 400, 3))

    assert score > 20


def test_score_haar_boxes_rejects_implausible_boxes() -> None:
    score = score_haar_boxes(
        [
            (10, 10, 4, 4),
            (10, 10, 300, 30),
            (10, 10, 280, 260),
        ],
        1.0,
        (300, 400, 3),
    )

    assert score == 0.0


def test_contrast_stretch_image_expands_dark_tonal_range() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, :10] = 20
    image[:, 10:] = 80

    stretched = contrast_stretch_image(image)

    assert stretched.min() == 0
    assert stretched.max() == 255
    assert stretched.shape == image.shape


def test_contrast_stretch_image_leaves_flat_images_valid() -> None:
    image = np.full((20, 20, 3), 42, dtype=np.uint8)

    stretched = contrast_stretch_image(image)

    assert np.array_equal(stretched, image)


def test_main_writes_numbered_outputs_for_split_scan(tmp_path: Path) -> None:
    image = np.full((180, 260, 3), 255, dtype=np.uint8)
    image[25:155, 20:105] = (45, 100, 150)
    image[25:155, 150:240] = (150, 90, 45)
    input_path = tmp_path / "scan.jpg"
    output_dir = tmp_path / "out"
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(input_path)

    exit_code = main([str(input_path), "-o", str(output_dir), "--skip-orientation"])

    assert exit_code == 0
    assert (output_dir / "scan_1.jpg").exists()
    assert (output_dir / "scan_2.jpg").exists()
    assert not (output_dir / "scan.jpg").exists()


def test_main_preflights_numbered_output_collisions(tmp_path: Path) -> None:
    image = np.full((180, 260, 3), 255, dtype=np.uint8)
    image[25:155, 20:105] = (45, 100, 150)
    image[25:155, 150:240] = (150, 90, 45)
    input_path = tmp_path / "scan.jpg"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(input_path)
    (output_dir / "scan_2.jpg").write_bytes(b"existing")

    exit_code = main([str(input_path), "-o", str(output_dir), "--skip-orientation"])

    assert exit_code == 1
    assert not (output_dir / "scan_1.jpg").exists()


def make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), "white").save(path)
