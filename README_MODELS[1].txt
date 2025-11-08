=== Models README ===
This enhanced package does NOT include pretrained model binaries due to licensing and size.
To run the system you should add an ONNX embedding model (e.g., ArcFace) and optionally detector weights.

Recommended steps (run locally):
1. Download a public ArcFace ONNX from model zoo or convert PyTorch to ONNX.
2. Place the file at: models/arcface.onnx
3. (Optional) Install insightface and let the detector wrapper download detector models automatically:
   pip install insightface onnxruntime opencv-python-headless
4. If you prefer a ready detector, download one from insightface model zoo and set DETECTOR_WEIGHTS env var.

If you want me to include a specific ONNX file, upload it here and I will place it in the ZIP.
