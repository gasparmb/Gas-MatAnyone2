# Gaspar's MatAnyone 2 — Custom Interface

A custom web interface for [MatAnyone2](https://github.com/pq-yang/MatAnyone), built on top of the original project. Features a dark UI with single-video and batch matting modes, bidirectional propagation, and a native desktop app wrapper via PyWebView.

## Features

- **Single video mode** — upload a video, annotate a subject with positive/negative clicks, launch matting
- **Batch mode** — queue multiple videos, annotate each one, process them all sequentially
- **Bidirectional propagation** — annotate any frame in the middle of a video; the model propagates both forward and backward
- **Multi-mask support** — annotate several subjects independently with color-coded masks
- **Resolution control** — downscale input before processing (Original / 1152p / 1080p / 720p / 540p / 480p)
- **Desktop app** — launch as a native macOS window via PyWebView (`launch.py`)
- **Native download dialog** — save results anywhere on disk via the system save dialog

## Requirements

- macOS (tested on Apple Silicon — MPS acceleration)
- Python 3.10+
- The base MatAnyone2 project installed (see below)

## Installation

### 1. Clone and set up MatAnyone2

Follow the original MatAnyone2 setup instructions to install the base project and download pretrained models into `pretrained_models/`.

### 2. Install additional dependencies

```bash
pip install fastapi uvicorn[standard] python-multipart psutil \
    imageio==2.25.0 "imageio[ffmpeg]" ffmpeg-python \
    opencv-python pywebview
```

### 3. Install the matanyone2 package (without reinstalling torch)

```bash
pip install -e . --no-deps
```

## Usage

### Desktop app (recommended)

```bash
cd hugging_face
python launch.py
```

Opens a native macOS window with the full interface.

### Web server only

```bash
cd hugging_face
uvicorn custom_server:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000` in your browser.

## Project structure

```
hugging_face/
├── custom_index.html     # UI (single file, no build step)
├── custom_server.py      # FastAPI backend
├── launch.py             # PyWebView desktop launcher
├── matanyone2_wrapper.py # Bidirectional matting logic
├── requirements.txt      # Dependencies
└── tools/                # Utilities from original MatAnyone project
```

## Credits

Built on top of [MatAnyone](https://github.com/pq-yang/MatAnyone) by pq-yang.
Custom interface by [Gaspar Matheron](https://gasparmatheron.studio).
