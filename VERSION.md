# Parts Hotspot v4.18 YOLO HQ Fragment Fix

Build: `2026-07-09 11:12 v4.18 YOLO HQ-fragment-fix`

This snapshot preserves the desktop version with:

- custom `models/yolo_numbers.pt` number detector;
- 800x800 work and export coordinates;
- high-quality OCR source retained separately from the 800x800 image;
- cropped PDF fragment OCR capped at 3200x3200;
- PDF-fragment OCR without the redundant OpenCV multiscale pass;
- per-backend OCR progress;
- batched YOLO candidate OCR;
- CPU-compatible PyTorch bootstrap in `run.bat`;
- OpenCV leader-line compatibility fix;
- guarded background OCR errors instead of an endless busy state.

Run the application with `run.bat`.
