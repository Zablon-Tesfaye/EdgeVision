"""Flask dashboard with offline camera streaming and status reporting."""

from threading import Lock
from time import perf_counter, sleep

import cv2
from flask import Flask, Response, jsonify, render_template

from .camera import annotate, open_camera
from .config import Settings
from .inference import ObjectDetector
from .health import readiness

app = Flask(__name__, template_folder="../templates", static_folder="../static")
settings = Settings()
_detector = None
_lock = Lock()
_state = {"frames": 0, "inference_ms": None, "fps": None, "detections": []}


def get_detector():
    global _detector
    if _detector is None:
        with _lock:
            if _detector is None:
                _detector = ObjectDetector(settings.model_path, settings.labels_path, settings.confidence)
    return _detector


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    with _lock:
        return jsonify(dict(_state))


def frames():
    detector = get_detector()
    with open_camera(settings.camera_index, settings.input_width, settings.input_height) as capture:
        last = perf_counter()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = detector.detect(frame)
            now = perf_counter()
            fps = 1.0 / max(now - last, 0.000001)
            last = now
            annotated = annotate(frame, result.detections)
            ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            with _lock:
                _state.update({
                    "frames": _state["frames"] + 1,
                    "inference_ms": round(result.inference_ms, 1),
                    "fps": round(fps, 1),
                    "detections": [item.as_dict() for item in result.detections],
                })
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            sleep(max(0, 1 / settings.max_fps - (perf_counter() - now)))


@app.get("/api/health")
def health():
    report = readiness(settings.model_path, settings.labels_path)
    return jsonify(report), 200 if report["ready"] else 503


@app.get("/video")
def video():
    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


def main():
    settings.validate()
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)


if __name__ == "__main__":
    main()
