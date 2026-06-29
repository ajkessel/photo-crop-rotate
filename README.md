# Photo Crop Rotate

Batch-process scanned photographs by cropping scanner borders and rotating each
image to the face orientation detected by MediaPipe. If a scan clearly contains
multiple photos separated by white scanner margin, the scan is split into
numbered output files before each photo is cropped and rotated.

A typical use case is when you have scanned old photographs with a scanner that
does not detect orientation or crop to the size of the image. You can put all
of those photos into a folder and run this script. Assuming there are faces in
the photos, it will make a best guess at the proper rotation for each image.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage

```bash
python photo_crop_rotate.py scans/ -o processed/ --recursive
```

The script accepts any mix of image files and image directories. It writes a
`process_report.csv` file in the output directory with the crop status, rotation
decision, number of detected faces, and any errors.

Useful options:

```bash
python photo_crop_rotate.py scans/ -o processed/ --debug-dir debug/
python photo_crop_rotate.py scans/ -o processed/ --no-split
python photo_crop_rotate.py scans/ -o processed/ --skip-orientation
python photo_crop_rotate.py scans/ -o processed/ --no-contrast-fallback
python photo_crop_rotate.py scans/ -o processed/ --no-haar-fallback
python photo_crop_rotate.py scans/ -o processed/ --format jpeg --jpeg-quality 95
```

When splitting is triggered, outputs are named with a numeric suffix, such as
`scan_1.jpg` and `scan_2.jpg`. Single-photo scans keep the original output name.

On the first orientation run, the default MediaPipe face detector model is
downloaded into `models/`. Use `--model path/to/model.tflite` to provide a
model explicitly. If MediaPipe finds no faces, the script first retries face
detection on a contrast-stretched copy for dark images, then uses OpenCV's Haar
face detector as a fallback for distant grayscale group photos.

## License

[BSD 2-Clause][LICENSE]
