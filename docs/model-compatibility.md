# Model compatibility and reproducible benchmarks

The inference engine accepts **SSD-style four-output detection models**, not bare MobileNetV2 image classifiers. Validate the tensor signature with `python -m edgevision.model_validation models/detect.tflite` before using a camera.

Supported input dtypes are uint8 and float32. Float32 inputs are currently normalized to 0–1; models requiring -1–1 or raw 0–255 must use the appropriate model-specific preprocessing. Indexed labels preserve numeric class IDs. Output tensors must be boxes, classes, scores, count in that order.

Class-aware NMS removes overlapping duplicates after score filtering. Accuracy and retained accuracy must be measured on a labeled dataset. Run Pi FPS benchmarks on the Pi, not on a desktop. The 20 FPS / 98% retention resume claims are not verified by CI.
