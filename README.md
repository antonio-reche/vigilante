# Vigilante

Vigilante is a small home-monitoring system for keeping an eye on your place while you are away. It is a single Python script that uses a webcam to detect people and sends you an alert through Telegram bot. You can also ask the bot for a live photo or a short video with `/photo` or `/video`.

It uses a small object detection model (tiny YOLO) that runs on almost any CPU, making it suitable for old hardware without a GPU. It should also work on any computer with a webcam. Detection stays on the device, while Telegram handles alerts and commands.

## What you need

- A computer with a webcam and internet
- Python 3.10 or newer
- A Telegram account
- *OPTIONAL:* [ffmpeg](https://ffmpeg.org/) on your PATH (only if you want to use the `/video` command)

On Debian or Ubuntu:

```bash
sudo apt install python3 python3-venv python3-pip ffmpeg
```

On macOS with Homebrew: `brew install ffmpeg`. On Windows, install ffmpeg and check that `ffmpeg` works in a terminal.

## How it works

1. OpenCV keeps reading the latest frame from the webcam in the background.

2. Every few seconds, a small object-detection model ([YOLOS-tiny](https://huggingface.co/hustvl/yolos-tiny)) checks that frame for a person. Here we only care about the `person` label. If you want to detect something else (a `dog`, a `cat`, etc.), you can extend it to any label the model already knows.

3. If a person walks into camera view, the model detects that and the bot sends you a photo on Telegram (with a cooldown so one visit is usually one alert).

4. A second thread listens for commands you send to the bot. Only people whose chat ID you put in the config can talk to it.

Checks are spaced out so an old CPU can keep up. With the default `WATCH_TIME_STEP = 3 seconds` plus the time the model takes to run, you get a look roughly every 4 to 6 seconds depending on your device. This is a presence alarm. It does not record all day.

### Telegram rate limits (read this before you tweak timings)

Telegram will refuse you if you talk to their API too often. The reply is HTTP 429, then you wait `retry_after` seconds. It will get worse if you keep trying, and the bot can stop working. Two easy ways to hit that limit with this code:

1. **Alert photos.** If someone stands in view too long, a naive loop would call `sendPhoto` on every positive detection, and Telegram would start ignoring the bot.

2. **Checking for commands.** A tight `getUpdates` loop (ask Telegram “any new messages?” over and over with no wait) floods the same API. If you write your own listener, don’t do that. This script doesn’t.

If you want to change timings, do not set `COOLDOWN` or `WATCH_TIME_STEP` near zero. If you add more people to `CHAT_IDS`, each alert sends another message and photo, so the limit is easier to hit.

What the script already does to stay under the cap:

- Alert only on a **new** appearance, and only if `COOLDOWN` seconds have passed since the last send (default 30).

- After `PRESENCE_TIMEOUT` seconds with no person (default 8), the next appearance can alert again. One detection should be only one photo and not a burst.

- Sends go on a **background queue** (max 5). The camera loop never waits on the network. If the queue is already full, extra alerts are dropped instead of stacking.

- If Telegram still returns 429, the sender sleeps `retry_after` and continues. You will see `rate limit, waiting ...` in the terminal.

- Command listening uses **long polling**: `getUpdates` with `timeout=30`, so Telegram holds the request until a message arrives or 30 seconds pass. That is a handful of checks per minute, not hundreds. `offset` skips messages already seen. On error it sleeps 3 seconds instead of retrying immediately.

The first run downloads the model from Hugging Face, so it will take more time. Later runs can reuse the cached model. The default it's a small model (that works surprisingly well), but if you decide to use a better or larger one, make sure you have enough space and CPU/GPU to run it!

## Setup

1. Clone this repo and create a virtual environment:

```bash
git clone https://github.com/antonio-reche/vigilante.git
cd vigilante
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Create a Telegram bot with [@BotFather](https://telegram.me/BotFather) (see the [docs](https://core.telegram.org/bots/tutorial)). It will give you a **bot token** (a long string, not a numeric “bot ID”). Treat that token like a password: anyone who has it can send messages as your bot!

3. Get your **chat ID**. Open a chat with your new bot, send any message, then talk to [@userinfobot](https://telegram.me/userinfobot) or inspect `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` and copy the `chat.id` number. If you want someone else to get the alerts too and be able to talk to the bot, they need to message the bot once, and then you add their chat ID as well.

4. Copy the example config and put your values in it. Don’t share it!

```bash
cp params.example.json params.json
```

Set `BOT_TOKEN` and `CHAT_IDS`. If you only want this for yourself, one chat ID is enough.

5. Point the webcam pointing at a place you want to watch and one you’re allowed to film (your own place). Plug it in, then:

```bash
python detection.py
```
That's it!

You should get a “camera is live” photo in Telegram. After that, walk into view once if you want to test an alert.


Default camera index is `0` (the first webcam). If you have several cameras and you want a different one, change `camera_index` in `WebcamStream` inside `detection.py`.

## Commands

If you want a snapshot or a clip of the current view, send these in the chat with the bot (they only work from a chat ID in `CHAT_IDS`):

| Command | What it does |
|---|---|
| `/photo` | Snapshot of the current frame |
| `/video` | Short clip (default 5 seconds) |
| `/video 10s` | Clip of 10 seconds (capped at 20 seconds by `VIDEO_MAX_SECONDS`, default 20, but can be extended/lowered) |

## Config (`params.json`)

| Key | Meaning |
|---|---|
| `BOT_TOKEN` | Token from BotFather |
| `CHAT_IDS` | People who get alerts and can send commands. If you want someone else included, add their chat ID here |
| `IMAGES_FOLDER` | Local folder for saved detection photos |
| `WATCH_TIME_STEP` | Seconds to wait between detection checks (default 3) |
| `DETECTION_THRESHOLD` | Minimum confidence to count as a person (default 0.65) Tweak this to avoid false positives.|
| `YOLO_MODEL` | Hugging Face model name (default `hustvl/yolos-tiny`) |
| `LANGUAGE` | Bot text: `en` or `es`. If you want Spanish messages, set `es` |
| `COOLDOWN` | Minimum seconds between alerts (default 30) |
| `PRESENCE_TIMEOUT` | Seconds with no detection before the next visit can alert again (default 8) |
| `VIDEO_SECONDS` | Default `/video` length |
| `VIDEO_MAX_SECONDS` | Maximum `/video` length |
| `VIDEO_FPS` | Frames per second of clips |

## Limits and privacy

- One camera. Slow cadence (about 4 to 6 seconds between checks). Occasional false positives (do tests and tweak the `DETECTION_THRESHOLD` parameter). It does not recognize faces.
- Telegram rate-limits bots. Leave `COOLDOWN` (30) and `WATCH_TIME_STEP` (3) alone unless you know you want to change them. Flooding `sendPhoto` **or** polling `getUpdates` in a tight loop gets you HTTP 429 and dropped alerts. Keep long polling (`timeout=30`).
- Alert photos and clips go through Telegram’s servers. That traffic is not end-to-end encrypted. Detection stays on your computer; the files you choose to send do not.
- Only people in `CHAT_IDS` can send `/photo` and `/video`. Keep `params.json` off git. If a token ever leaked, revoke it in BotFather with `/revoke` and put the new token in `params.json`.
- This is a personal project I built so I could check inside my home when I'm not around, so, if you want to use this repo, point the camera only at your own space. Recording laws differ by country and by whether the lens can see a neighbor, a sidewalk, or a shared hallway.
