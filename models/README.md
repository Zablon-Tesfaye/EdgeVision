# Model assets

Model weights are not committed. Place a compatible SSD-style TensorFlow Lite model at `models/detect.tflite` and its class names at `models/labels.txt`.

The inference adapter expects:
- Input: one RGB image, NHWC layout, `uint8` or `float32`
- Outputs in order: normalized boxes `[1,N,4]` in `ymin,xmin,ymax,xmax` order, class IDs `[1,N]`, scores `[1,N]`, detection count `[1]`
- Class IDs matching the label file indexing

Not all MobileNetV2 models use this signature. A classification MobileNetV2 model **cannot** produce bounding boxes. Use an SSD MobileNetV2 detection model with compatible postprocessing or adapt `edgevision/inference.py` for your model's tensor signature.

Model licenses vary. Review the license before distributing weights.
