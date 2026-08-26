import base64
from threading import Lock, Thread
import queue
import cv2
from cv2 import VideoCapture, imencode
import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
import time
import requests
from datetime import datetime
import json
import os
import subprocess
import uuid


def open_json(path):
    """Read a JSON file and return its contents."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


## LOAD PARAMETERS ===============================
PARAMS = open_json("params.json")

# Required parameters to be manually set in params.json
BOT_TOKEN = PARAMS["BOT_TOKEN"]                  # Telegram bot token (given by Telegram when creating the bot with @BotFather)
CHAT_IDS = PARAMS["CHAT_IDS"]                    # Chat IDs that get alerts and can send commands to the bot

# Extra parameters (with defaults if missing from params.json)
IMAGES_FOLDER = PARAMS.get("IMAGES_FOLDER", "images")              # Folder for saved detection photos
WATCH_TIME_STEP = PARAMS.get("WATCH_TIME_STEP", 3)                 # Seconds between each detection check
DETECTION_THRESHOLD = PARAMS.get("DETECTION_THRESHOLD", 0.65)      # Min confidence to count as a person
YOLO_MODEL = PARAMS.get("YOLO_MODEL", "hustvl/yolos-tiny")         # Name of the detection model
LANGUAGE = PARAMS.get("LANGUAGE", "en")                            # "en" or "es" for bot messages
COOLDOWN = PARAMS.get("COOLDOWN", 30)                              # Min seconds between alerts
PRESENCE_TIMEOUT = PARAMS.get("PRESENCE_TIMEOUT", 8)               # Seconds with no detection before "left"
VIDEO_SECONDS = PARAMS.get("VIDEO_SECONDS", 5)                     # Default /video clip length
VIDEO_MAX_SECONDS = PARAMS.get("VIDEO_MAX_SECONDS", 20)            # Cap for /video Xs
VIDEO_FPS = PARAMS.get("VIDEO_FPS", 8)                             # Frames per second of /video clips
# ================================================

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}" # Telegram API URL to interact with the bot


class WebcamStream:
    """Camera reader that always keeps the latest frame in a background thread.

    Without this, OpenCV would hand us a buffered (old) image while detection is busy.
    """
    def __init__(self, camera_index=0):
        """Open the camera. Index 0 is usually the default webcam."""
        self.stream = VideoCapture(index=camera_index)
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()

    def start(self):
        """Start grabbing frames in the background. Safe to call more than once."""
        if self.running:
            return self
        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        """Loop: read from the camera and store the newest frame."""
        while self.running:
            _, frame = self.stream.read()
            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    def read(self, encode=False):
        """Return a copy of the latest frame. If encode is True, return it as JPEG in base64."""
        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()
        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)
        return frame

    def stop(self):
        """Stop the background thread."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def end(self):
        """Stop the thread and free the camera."""
        self.stop()
        self.stream.release()


def is_person(results):
    """Return (True, alert text) if a person was detected, otherwise (False, None)."""
    if LANGUAGE == "es":
        person_message = "⚠️⚠️ ADVERTENCIA ⚠️⚠️\n¡Persona detectada!"
    else:
        person_message = "⚠️⚠️ WARNING ⚠️⚠️\nPerson detected!"

    for _, label, _ in zip(results["scores"], results["labels"], results["boxes"]):
        detection = model.config.id2label[label.item()]
        if detection == "person":
            return True, person_message
    return False, None


def draw_person_boxes(image, results):
    """Draw a box and confidence label around each detected person. Returns the same image."""
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detection = model.config.id2label[label.item()]
        if detection == "person":
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 215, 255), 2)
            confidence_text = f"person ({round(score.item() * 100)}%)"
            text_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(image, (x1, y1 - 25), (x1 + text_size[0] + 10, y1), (255, 255, 255), -1)
            cv2.putText(image, confidence_text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    return image


# ============ BACKGROUND SENDING ============
# The detection loop NEVER blocks waiting for Telegram (important
# on a slow connection). Sends go to a queue and
# a separate thread processes them one by one, respecting 429s.
send_queue = queue.Queue(maxsize=5)


def _post_alert(chat_id, message, photo_bytes):
    """Send a text message and a photo to one Telegram chat. Waits if Telegram rate-limits us."""
    try:
        if message:
            requests.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
                          timeout=20)
        resp = requests.post(f"{TELEGRAM_API}/sendPhoto",
                             files={"photo": ("deteccion.jpg", photo_bytes, "image/jpeg")},
                             data={"chat_id": chat_id}, timeout=30)
        if resp.status_code == 429:
            retry = resp.json().get("parameters", {}).get("retry_after", 5)
            print(f"[telegram] rate limit, espero {retry}s")
            time.sleep(retry)
    except Exception as e:
        print("[telegram] error enviando:", e)


def telegram_worker():
    """Take alerts from the queue and send them one by one to every trusted chat."""
    while True:
        message, photo_bytes = send_queue.get()
        for chat_id in CHAT_IDS:
            _post_alert(chat_id, message, photo_bytes)
        send_queue.task_done()


def enqueue_alert(message, frame):
    """Put an alert on the send queue. If the queue is full, drop it instead of piling up."""
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return
    try:
        send_queue.put_nowait((message, buf.tobytes()))
    except queue.Full:
        # There's already an alert in the queue: drop it instead of stacking.
        pass


Thread(target=telegram_worker, daemon=True).start()


# ============ SHORT CLIP (/video) ============
# Records a few seconds of the stream, compresses with ffmpeg (H.264, no audio)
# and leaves a small mp4, playable in Telegram.
def parse_video_seconds(text):
    """Read a /video command and return how many seconds to record (default, or the number given, capped)."""
    parts = text.split()
    if len(parts) < 2:
        return VIDEO_SECONDS
    raw = parts[1].lower().rstrip("s")
    try:
        seconds = int(float(raw))
    except ValueError:
        return VIDEO_SECONDS
    if seconds < 1:
        seconds = 1
    if seconds > VIDEO_MAX_SECONDS:
        seconds = VIDEO_MAX_SECONDS
    return seconds


def record_short_clip(seconds):
    """Record a short camera clip, compress it with ffmpeg, and return the path to the mp4 file."""
    tag = uuid.uuid4().hex
    raw_path = f"/tmp/clip_{tag}.avi"
    mp4_path = f"/tmp/clip_{tag}.mp4"

    frame = webcam.read()
    h, w = frame.shape[:2]
    if w > 640:
        h = int(h * 640 / w)
        w = 640
    w -= w % 2
    h -= h % 2

    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"MJPG"), VIDEO_FPS, (w, h))
    if not writer.isOpened():
        raise RuntimeError("No se pudo abrir el VideoWriter")

    interval = 1.0 / VIDEO_FPS
    t0 = time.time()
    try:
        for i in range(int(seconds * VIDEO_FPS)):
            f = webcam.read()
            if f.shape[1] != w or f.shape[0] != h:
                f = cv2.resize(f, (w, h))
            writer.write(f)
            delay = (i + 1) * interval - (time.time() - t0)
            if delay > 0:
                time.sleep(delay)
        writer.release()
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_path,
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "32",
             "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
             mp4_path],
            check=True, capture_output=True,
        )
    finally:
        writer.release()
        if os.path.exists(raw_path):
            os.remove(raw_path)
    return mp4_path


def handle_video(chat_id, seconds):
    """Tell the chat that recording started, record the clip, send it, then delete the temp file."""
    try:
        text = f"🎥 Grabando {seconds}s..." if LANGUAGE == "es" else f"🎥 Recording {seconds}s..."
        requests.post(f"{TELEGRAM_API}/sendMessage",
                      data={"chat_id": chat_id, "text": text}, timeout=20)
        path = record_short_clip(seconds)
        try:
            with open(path, "rb") as f:
                requests.post(f"{TELEGRAM_API}/sendVideo",
                              files={"video": ("actual.mp4", f, "video/mp4")},
                              data={"chat_id": chat_id}, timeout=60)
        finally:
            if os.path.exists(path):
                os.remove(path)
    except Exception as e:
        print("[telegram] error enviando vídeo:", e)


# ============ COMMAND LISTENER (/photo, /video) ============
# Long polling: getUpdates with timeout=30. Telegram keeps the connection
# open until a message arrives or 30s pass. This is NOT a tight loop,
# so it won't flood or lock up. The 'offset' avoids rereading old messages.
def listen_commands():
    """Listen for /photo and /video from trusted chats using Telegram long polling."""
    # In case a webhook was set at some point (it would clash with getUpdates).
    try:
        requests.get(f"{TELEGRAM_API}/deleteWebhook", timeout=10)
    except Exception:
        pass

    offset = None
    allowed = {str(c) for c in CHAT_IDS}
    while True:
        try:
            params = {"timeout": 30}
            if offset is not None:
                params["offset"] = offset
            r = requests.get(f"{TELEGRAM_API}/getUpdates", params=params, timeout=40)
            for upd in r.json().get("result", []):
                offset = upd["update_id"] + 1
                msg = upd.get("message", {})
                text = msg.get("text") or ""
                chat = str(msg.get("chat", {}).get("id", ""))
                # Only reply to trusted chats.
                if chat not in allowed:
                    continue
                if text.startswith("/photo"):
                    frame = webcam.read()
                    ok, buf = cv2.imencode(".jpg", frame)
                    if ok:
                        requests.post(f"{TELEGRAM_API}/sendPhoto",
                                      files={"photo": ("actual.jpg", buf.tobytes(), "image/jpeg")},
                                      data={"chat_id": chat}, timeout=30)
                elif text.startswith("/video"):
                    seconds = parse_video_seconds(text)
                    Thread(target=handle_video, args=(chat, seconds), daemon=True).start()
        except Exception as e:
            print("[telegram] error escuchando comandos:", e)
            time.sleep(3)


Thread(target=listen_commands, daemon=True).start()


# ============ STARTUP ============
model = YolosForObjectDetection.from_pretrained(YOLO_MODEL)
image_processor = YolosImageProcessor.from_pretrained(YOLO_MODEL)

webcam = WebcamStream().start()

print("\n============= START WATCHING =============\n")

if LANGUAGE == "es":
    init_message = (
        "👋 ¡Hola! 👋\n"
        "La cámara acaba de activarse! 📹\n"
        "A partir de ahora, te avisaré si detecto a alguien.\n\n"
        "Comandos:\n"
        "• /photo -> tomo una imagen actual\n"
        "• /video -> grabo un clip corto\n"
        f"• /video 10s -> grabo un clip de 10s (máx. {VIDEO_MAX_SECONDS}s)"
    )
else:
    init_message = (
        "👋 Hello there! 👋\n"
        "Security cam is live! 📹\n"
        "I'll alert you if I spot a person.\n\n"
        "Commands:\n"
        "• /photo -> I take a current image\n"
        "• /video -> I record a short clip\n"
        f"• /video 10s -> I record a 10s clip (max {VIDEO_MAX_SECONDS}s)"
    )

init_photo = webcam.read()
cv2.putText(init_photo, "CAMERA IS LIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
enqueue_alert(init_message, init_photo)


# ============ MAIN LOOP ============
last_sent = 0.0
last_detection = 0.0
person_present = False

while True:
    try:
        frame = webcam.read()
        input_image = image_processor(images=frame, return_tensors="pt")
        outputs = model(**input_image)
        target_sizes = torch.tensor([frame.shape[:-1]])
        results = image_processor.post_process_object_detection(
            outputs, threshold=DETECTION_THRESHOLD, target_sizes=target_sizes)[0]

        detected, message = is_person(results)
        now = time.time()

        if detected:
            last_detection = now
            # Only alert if this is a NEW appearance and the cooldown has passed.
            if not person_present and (now - last_sent) > COOLDOWN:
                annotated = draw_person_boxes(frame.copy(), results)
                enqueue_alert(message, annotated)
                last_sent = now

                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join(IMAGES_FOLDER, "image_" + current_time + ".png"), annotated)
                print("Persona detectada! 🚨")
            person_present = True
        else:
            # If nobody has been seen for a while, reset so the next
            # appearance can alert again.
            if person_present and (now - last_detection) > PRESENCE_TIMEOUT:
                person_present = False

    except Exception as e:
        print("Error when running detection algorithm.")
        print(e)

    time.sleep(WATCH_TIME_STEP)
