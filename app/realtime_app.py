import cv2
import time
import logging
import threading
import numpy as np
import tensorflow as tf
from pathlib import Path
from queue import Queue

# MediaPipe with modern Task API fallback
mp_face = None
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    mp_tasks = True
except (ImportError, AttributeError):
    mp_tasks = False

from src.config import (
    TFLITE_QUANT_PATH, IMG_SIZE, CLASS_NAMES, Colors,
    FACE_DETECTION_CONFIDENCE, TFLITE_NUM_THREADS, SCREENSHOTS_DIR
)
from app.hud_utils import draw_hud

logger = logging.getLogger("RealtimeApp")


class VideoStream:
    """Threaded video stream for high FPS capture."""
    def __init__(self, camera_id=0):
        self.stream = cv2.VideoCapture(camera_id)
        if not self.stream.isOpened():
            raise RuntimeError(
                f"Could not open camera (id={camera_id}). "
                "Check that a webcam is connected and not in use by another application."
            )
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                self.stopped = True
                continue
            self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


class MaskDetector:
    def __init__(self, model_path=None):
        self.model_path = model_path or str(TFLITE_QUANT_PATH)
        logger.info(f"Initializing MaskDetector with: {self.model_path}")

        # Load Mask Model
        self.interpreter = tf.lite.Interpreter(
            model_path=self.model_path,
            num_threads=TFLITE_NUM_THREADS
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Face Detection Setup
        self.setup_detector()

    def setup_detector(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        logger.info("Using Optimized Haar Cascade for face detection.")

    def preprocess(self, face_img):
        face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        face_img = face_img.astype(np.float32)
        face_img = (face_img / 127.5) - 1.0
        return np.expand_dims(face_img, axis=0)

    def predict(self, face_img):
        start_time = time.time()
        input_data = self.preprocess(face_img)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]

        label = "Mask" if prediction >= 0.5 else "No Mask"
        confidence = prediction if prediction >= 0.5 else 1.0 - prediction
        latency = time.time() - start_time
        return label, float(confidence), latency


def run_realtime(model_path=None, camera_id=0, show_gradcam=False):
    try:
        detector = MaskDetector(model_path)
        vs = VideoStream(camera_id).start()
    except RuntimeError as e:
        logger.error(str(e))
        return

    logger.info("Professional Real-time detection launched. Press Q to quit.")

    paused = False
    fps_history = []
    latency_history = []
    prev_time = time.time()
    avg_fps = 0.0
    final_latency = 0.0
    display_frame = None

    # Window setup
    cv2.namedWindow("Mask Detection Pro G11", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mask Detection Pro G11", 1280, 720)

    while not vs.stopped:
        curr_time = time.time()
        time_diff = curr_time - prev_time
        if time_diff > 0:
            fps = 1.0 / time_diff
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
        prev_time = curr_time

        if not paused:
            frame = vs.read()
            if frame is None:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # 1. Faster Face Detection (Downscale search)
            search_scale = 0.5
            small_gray = cv2.cvtColor(
                cv2.resize(frame, (0, 0), fx=search_scale, fy=search_scale),
                cv2.COLOR_BGR2GRAY
            )
            # Tighter params: scaleFactor=1.1, minNeighbors=6 → fewer false positives
            faces = detector.face_cascade.detectMultiScale(
                small_gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40)
            )

            labels, bboxes, confs = [], [], []
            total_latency = 0

            for (bx, by, bw, bh) in faces:
                # Upscale coordinates back to original frame size
                bx = int(bx / search_scale)
                by = int(by / search_scale)
                bw = int(bw / search_scale)
                bh = int(bh / search_scale)

                # Expand ROI slightly for better mask context
                pad = 15
                bx_p = max(0, bx - pad)
                by_p = max(0, by - pad)
                bw_p = min(w - bx_p, bw + 2 * pad)
                bh_p = min(h - by_p, bh + 2 * pad)

                face_roi = frame[by_p:by_p + bh_p, bx_p:bx_p + bw_p]
                if face_roi.size > 0:
                    label, conf, lat = detector.predict(face_roi)
                    labels.append(label)
                    bboxes.append((bx, by, bw, bh))
                    confs.append(conf)
                    total_latency += lat

            if labels:
                avg_latency = total_latency / len(labels)
                latency_history.append(avg_latency)
                if len(latency_history) > 30:
                    latency_history.pop(0)
                final_latency = sum(latency_history) / len(latency_history)
            else:
                final_latency = final_latency  # keep last value for smooth HUD

            # 2. Render HUD
            display_frame = draw_hud(
                frame.copy(),
                labels, bboxes, confs,
                fps=avg_fps,
                status="ACTIVE" if not paused else "PAUSED",
                inference_time=final_latency,
                face_count=len(labels),
            )

            cv2.imshow("Mask Detection Pro G11", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            logger.info("Detection %s.", "paused" if paused else "resumed")
        elif key == ord('s'):
            if display_frame is not None:
                filename = f"screenshot_{int(time.time())}.png"
                filepath = SCREENSHOTS_DIR / filename
                cv2.imwrite(str(filepath), display_frame)
                logger.info(f"Screenshot saved: {filepath}")
            else:
                logger.warning("No frame available to screenshot yet.")

    vs.stop()
    cv2.destroyAllWindows()
    logger.info("Detection stopped.")


if __name__ == "__main__":
    run_realtime()
