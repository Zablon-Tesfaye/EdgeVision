# Resume evidence checklist

The offline TFLite/OpenCV/Flask pipeline and local dashboard are implemented. The model is user-supplied.

Before making quantitative claims:

1. Inspect the chosen MobileNetV2 SSD detector with `python -m edgevision.model_validation models/detect.tflite` and confirm its output tensor signature matches the adapter.
2. Compare actual original and quantized file sizes with `python -m edgevision.model_comparison ORIGINAL QUANTIZED`.
3. Run `python -m edgevision.benchmark --model models/detect.tflite --labels models/labels.txt --image SAMPLE.jpg` **on Raspberry Pi 4**. This reports estimated still-image throughput, not camera FPS.
4. Record observed live camera FPS separately, including resolution, threads, device, and model version.
5. Evaluate both models on the same labeled detection dataset using mAP or another stated metric, then calculate retained accuracy.

The resume figures (14 MB to 3.5 MB, 20 FPS, 98% accuracy retention) are **not yet verified**. Never substitute file-size reduction for measured accuracy or desktop FPS for Pi FPS.
