# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

import cv2
from PyInstaller.utils.hooks import collect_all


datas = []
binaries = []
hiddenimports = []

model_path = Path("models") / "blaze_face_full_range.tflite"
if model_path.exists():
    datas.append((str(model_path), "models"))

datas.append((cv2.data.haarcascades, "cv2/data"))

mediapipe_datas, mediapipe_binaries, mediapipe_hiddenimports = collect_all("mediapipe")
datas += mediapipe_datas
binaries += mediapipe_binaries
hiddenimports += mediapipe_hiddenimports

a = Analysis(
    ["photo_crop_rotate.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="photo-crop-rotate",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
