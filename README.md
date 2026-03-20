# Gaspar's MatAnyone 2

A custom desktop interface for [MatAnyone 2](https://github.com/pq-yang/MatAnyone2), built on top of the original project by pq-yang. Replaces the Gradio demo with a dark web UI wrapped in a native macOS app via PyWebView.

## Features

- **Single video mode** — upload, annotate with clicks, launch matting
- **Batch mode** — queue multiple videos, annotate each, process sequentially with a download-all button
- **Bidirectional propagation** — annotate any frame; the model propagates forward and backward from that point
- **Multi-mask support** — several subjects per video, color-coded chips
- **Resolution control** — Original / 1152p / 1080p / 720p / 540p / 480p
- **Native macOS app** — launches as a desktop window via PyWebView
- **Native save dialog** — choose where to save results on disk

## Installation

### 1. Set up the base project

```bash
git clone https://github.com/gasparmatheron/Gas-MatAnyone2
cd Gas-MatAnyone2
conda create -n matanyone2 python=3.10 -y
conda activate matanyone2
pip install -e . --no-deps
```

The pretrained model downloads automatically on first launch. Or download manually from [MatAnyone2 releases](https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth) into `pretrained_models/`.

### 2. Install interface dependencies

```bash
pip install fastapi uvicorn[standard] python-multipart psutil \
    imageio==2.25.0 "imageio[ffmpeg]" ffmpeg-python \
    opencv-python matplotlib pywebview
pip install -r hugging_face/requirements.txt
```

## Usage

```bash
cd hugging_face
python launch.py
```

Or as a web server only:

```bash
uvicorn custom_server:app --host 127.0.0.1 --port 8000
```

## Project structure

```
hugging_face/
├── custom_index.html     # UI (single file, no build step)
├── custom_server.py      # FastAPI backend
├── launch.py             # PyWebView desktop launcher
├── matanyone2_wrapper.py # Bidirectional matting logic
└── tools/                # Utilities from original MatAnyone project
```

## Credits

- **MatAnyone 2** — [pq-yang](https://github.com/pq-yang/MatAnyone2), S-Lab NTU
- **Custom interface** — [Gaspar Matheron](https://gasparmatheron.studio)

## License

Based on [MatAnyone 2](https://github.com/pq-yang/MatAnyone2), licensed under [NTU S-Lab License 1.0](./LICENSE).
