# EdgeVision

**Private, offline-first object detection** with Python, OpenCV, TensorFlow Lite, and a lightweight Flask monitoring dashboard. Built for Raspberry Pi 4 experimentation and desktop development.

## What it does

- Reads frames from a local camera and runs inference without cloud calls
- Decodes SSD-style MobileNet detection outputs into labeled bounding boxes
- Streams annotated frames to a responsive local dashboard
- Shows measured inference latency, observed camera FPS, and detected objects
- Includes a still-image CLI, quantization script, benchmarking tool, and unit tests

No external API keys or accounts are required. **A compatible model and label file must be provided separately**; the project does not ship copyrighted model weights.

## Quick start

Use Python 3.11 on a laptop or Raspberry Pi.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Desktop only, if you do not have tflite-runtime:
pip install tensorflow
```

For Raspberry Pi, install a compatible `tflite-runtime` wheel instead of full TensorFlow. Wheel availability depends on Python version and architecture. On Pi, system OpenCV may be easier to install than the pip wheel.

Download a **TFLite SSD object detection** model compatible with the output signature described in [models/README.md](models/README.md). Put it at `models/detect.tflite` and put one label per line in `models/labels.txt`.

```bash
python -m edgevision.web
```

Open **http://127.0.0.1:5000**. The page loads without a camera, but video streaming requires a working camera and model.

## Detect an image

```bash
python -m scripts.detect_image photo.jpg --output results/annotated.jpg
```

## Benchmark inference on your hardware

```bash
python -m edgevision.benchmark --model models/detect.tflite --labels models/labels.txt --image photo.jpg --runs 100 --warmup 10 --output results/benchmark.json
```

The benchmark reports model size, mean end-to-end latency, and estimated single-image throughput. Live camera FPS is measured separately in the dashboard.

## Quantize a model

The optional conversion script takes a TensorFlow SavedModel and calibration images:

```bash
pip install tensorflow
python scripts/quantize.py --saved-model path/to/saved_model --calibration-images path/to/calibration_images --output models/detect.tflite
```

Quantization compatibility depends on the model's operators and input normalization. The script does not guarantee an SSD-compatible four-output signature. Test the converted model before using it.

## Tests

```bash
pytest -q
```

Unit tests use synthetic output tensors and do not need camera hardware or model weights.

## Configuration

| Environment variable | Default | Description |
| --- | --- | --- |
| `EDGEVISION_MODEL` | `models/detect.tflite` | TFLite model path |
| `EDGEVISION_LABELS` | `models/labels.txt` | Class labels |
| `EDGEVISION_CAMERA` | `0` | OpenCV camera index |
| `EDGEVISION_CONFIDENCE` | `0.45` | Minimum detection confidence |

## Honest performance reporting

Claims such as **20 FPS**, **98% accuracy**, or **14 MB to 3.5 MB** are *not verified by this repository*. Performance depends on the model, test dataset, Raspberry Pi hardware, camera, and runtime. The benchmark tool records actual measurements when run on the target hardware. Accuracy requires a labeled evaluation dataset and a defined metric such as mAP; model size requires comparing actual artifact sizes.

## Architecture

```text
edgevision/
  config.py       Environment configuration
  inference.py    TFLite interpreter and SSD decoding
  camera.py       Camera lifecycle and bounding-box rendering
  types.py        Typed detection results
  metrics.py      Measurement utilities
  benchmark.py    Reproducible latency benchmark
  web.py          Flask dashboard and MJPEG stream
templates/        Dashboard HTML
static/           Responsive styles
scripts/          Model quantization and image inference
tests/            Hardware-independent tests
models/           User-supplied model and label assets
```

The Flask development server binds to localhost by default. Do not expose the unauthenticated camera stream to a public network.
