# YOLO models

Put a custom parts-number detection model here:

```text
models/yolo_numbers.pt
```

The app will use this file automatically. You can also point to another model with:

```text
set PARTS_YOLO_MODEL=C:\path\to\model.pt
```

Do not commit large `.pt` model files unless you intentionally want to store model weights in the repository.
