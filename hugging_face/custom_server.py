import sys
import os

# Allow MPS to use full memory (no pool cap). We manually synchronize + release
# after each frame in the wrapper, so the pool doesn't accumulate between frames.
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')

# Adjust paths like the original app.py
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import uuid
import time
import threading
import base64
import io
import json
import numpy as np
import cv2
import psutil
import torch
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from tools.painter import mask_painter
from tools.interact_tools import SamControler
from tools.misc import get_device
from tools.download_util import load_file_from_url
from matanyone2_wrapper import matanyone2 as matanyone2_fn
from matanyone2.utils.get_default_model import get_matanyone2_model
from matanyone2.inference.inference_core import InferenceCore
from hydra.core.global_hydra import GlobalHydra

import ffmpeg
import imageio

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE = str(get_device())
BASE_DIR = os.path.dirname(__file__)
CHECKPOINT_FOLDER = os.path.join(BASE_DIR, '..', 'pretrained_models')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
MATANYONE2_URL = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"

# ── Globals ───────────────────────────────────────────────────────────────────

_sam = None
_matanyone_model = None
sessions: dict = {}
jobs: dict = {}
batches: dict = {}

# ── Model helpers ─────────────────────────────────────────────────────────────

def get_sam():
    global _sam
    if _sam is None:
        print("Loading SAM model…")
        ckpt = load_file_from_url(SAM_URL, CHECKPOINT_FOLDER)
        # SAM runs on CPU to keep MPS free for MatAnyone2 inference
        _sam = SamControler(ckpt, 'vit_h', 'cpu')
        print("SAM ready.")
    return _sam


def get_matanyone():
    global _matanyone_model
    if _matanyone_model is None:
        print("Loading MatAnyone2 model…")
        ckpt = load_file_from_url(MATANYONE2_URL, CHECKPOINT_FOLDER)
        try:
            GlobalHydra.instance().clear()
        except Exception:
            pass
        _matanyone_model = get_matanyone2_model(ckpt, DEVICE)
        _matanyone_model = _matanyone_model.to(DEVICE).eval()
        print("MatAnyone2 ready.")
    return _matanyone_model

# ── Utilities ─────────────────────────────────────────────────────────────────

def frame_to_b64(frame) -> str:
    """Convert numpy RGB frame or PIL Image to base64 JPEG."""
    if isinstance(frame, np.ndarray):
        img = Image.fromarray(frame.astype(np.uint8))
    else:
        img = frame.convert('RGB') if frame.mode != 'RGB' else frame
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def resize_frames(frames: list, max_short_side: int) -> list:
    """Resize frames so the shorter side equals max_short_side."""
    h, w = frames[0].shape[:2]
    short = min(h, w)
    if short <= max_short_side:
        return frames
    scale = max_short_side / short
    new_w = int(w * scale)
    new_h = int(h * scale)
    # Ensure even dimensions (required by some codecs)
    new_w = new_w if new_w % 2 == 0 else new_w - 1
    new_h = new_h if new_h % 2 == 0 else new_h - 1
    return [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames]


def extract_frames(video_path: str):
    frames = []
    audio_path = ""
    try:
        audio_path = os.path.splitext(video_path)[0] + '_audio.wav'
        (ffmpeg
         .input(video_path)
         .output(audio_path, format='wav', acodec='pcm_s16le', ac=2, ar='44100')
         .run(overwrite_output=True, quiet=True))
    except Exception as e:
        print(f"Audio extraction error: {e}")
        audio_path = ""

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if psutil.virtual_memory().percent > 90:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps, audio_path


def add_audio_to_video(video_path, audio_path, output_path):
    try:
        (ffmpeg
         .output(ffmpeg.input(video_path), ffmpeg.input(audio_path),
                 output_path, vcodec='copy', acodec='aac')
         .run(overwrite_output=True, capture_stdout=True, capture_stderr=True))
        return output_path
    except Exception as e:
        print(f"FFmpeg audio merge error: {e}")
        return video_path


def generate_video(frames, output_path, fps=30, gray2rgb=False, audio_path=""):
    frames_np = np.asarray(frames)
    if gray2rgb:
        frames_np = np.repeat(frames_np, 3, axis=3)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp = output_path.replace('.mp4', '_temp.mp4')
    _, h, w, _ = frames_np.shape
    imageio.mimwrite(temp, frames_np, fps=fps, quality=7,
                     codec='libx264', ffmpeg_params=["-vf", f"scale={w}:{h}"])
    if audio_path and os.path.exists(audio_path):
        result = add_audio_to_video(temp, audio_path, output_path)
        if result == output_path and os.path.exists(temp):
            os.remove(temp)
        return result
    return temp


def build_template_mask(sess, selected_names):
    multi = sess['multi_mask']['masks']
    names = sess['multi_mask']['mask_names']
    if not multi:
        return sess['masks'][sess['select_frame_number']]
    if not selected_names:
        selected_names = names[:]
    selected_sorted = sorted(selected_names)
    idx0 = int(selected_sorted[0].split('_')[1]) - 1
    tmpl = multi[idx0] * (idx0 + 1)
    for s in selected_sorted[1:]:
        i = int(s.split('_')[1]) - 1
        tmpl = np.clip(tmpl + multi[i] * (i + 1), 0, i + 1)
    return tmpl

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Pydantic models ───────────────────────────────────────────────────────────

class ClickReq(BaseModel):
    session_id: str
    x_norm: float
    y_norm: float
    is_positive: bool = True

class SessionReq(BaseModel):
    session_id: str

class SelectFrameReq(BaseModel):
    session_id: str
    frame_index: int

class RunReq(BaseModel):
    session_id: str
    erode: int = 10
    dilate: int = 10
    selected_masks: List[str] = []
    max_mem_frames: int = 5

class BatchRunReq(BaseModel):
    items: List[RunReq]

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(BASE_DIR, 'custom_index.html')
    with open(html_path, encoding='utf-8') as f:
        return f.read()


@app.post("/upload")
async def upload(file: UploadFile = File(...), max_size: Optional[str] = Form(None)):
    session_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or 'video.mp4')[1] or '.mp4'
    video_path = os.path.join(UPLOADS_DIR, session_id + ext)

    with open(video_path, 'wb') as f:
        f.write(await file.read())

    frames, fps, audio = extract_frames(video_path)
    if not frames:
        raise HTTPException(400, "Could not extract frames from video.")

    # Resize if requested
    if max_size and max_size != 'original':
        try:
            frames = resize_frames(frames, int(max_size))
        except Exception as e:
            print(f"Resize error: {e}")

    sam = get_sam()
    sam.sam_controler.reset_image()
    sam.sam_controler.set_image(frames[0])

    h, w = frames[0].shape[:2]
    sessions[session_id] = {
        'video_name': file.filename,
        'video_path': video_path,
        'origin_images': frames,
        'painted_images': [f.copy() for f in frames],
        'masks': [np.zeros((h, w), np.uint8)] * len(frames),
        'logits': [None] * len(frames),
        'select_frame_number': 0,
        'fps': fps,
        'audio': audio,
        'click_state': [[], []],
        'multi_mask': {'mask_names': [], 'masks': []},
    }

    return {
        'session_id': session_id,
        'total_frames': len(frames),
        'fps': fps,
        'image_w': w,
        'image_h': h,
        'video_name': file.filename,
        'frame': frame_to_b64(frames[0]),
    }


@app.post("/sam-click")
def sam_click(req: ClickReq):
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found")
    sess = sessions[req.session_id]
    fi = sess['select_frame_number']
    frame = sess['origin_images'][fi]
    h, w = frame.shape[:2]
    x = int(req.x_norm * w)
    y = int(req.y_norm * h)

    cs = sess['click_state']
    cs[0].append([x, y])
    cs[1].append(1 if req.is_positive else 0)

    sam = get_sam()
    sam.sam_controler.reset_image()
    sam.sam_controler.set_image(frame)

    prompt = {
        'point_coords': np.array(cs[0]),
        'point_labels': np.array(cs[1]),
        'multimask_output': False,
    }
    masks, scores, logits = sam.sam_controler.predict(prompt, 'point', False)
    best = np.argmax(scores)
    mask, logit = masks[best], logits[best]

    # Second pass with mask hint if negative click present
    if 0 in cs[1]:
        prompt2 = {**prompt, 'mask_input': logit[None]}
        masks2, scores2, logits2 = sam.sam_controler.predict(prompt2, 'both', False)
        best2 = np.argmax(scores2)
        mask, logit = masks2[best2], logits2[best2]

    sess['masks'][fi] = mask
    sess['logits'][fi] = logit

    from tools.painter import mask_painter, point_painter
    painted = mask_painter(frame, mask.astype('uint8'), 3, 0.7, 2, 5)
    pos_pts = np.array(cs[0])[np.array(cs[1]) > 0]
    neg_pts = np.array(cs[0])[np.array(cs[1]) < 1]
    if len(pos_pts): painted = point_painter(painted, pos_pts, 50, 0.9, 15, 2, 5)
    if len(neg_pts): painted = point_painter(painted, neg_pts, 8,  0.9, 15, 2, 5)
    painted_img = Image.fromarray(painted)
    sess['painted_images'][fi] = painted_img

    return {'frame': frame_to_b64(painted_img)}


@app.post("/add-mask")
def add_mask(req: SessionReq):
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found")
    sess = sessions[req.session_id]
    fi = sess['select_frame_number']

    sess['multi_mask']['masks'].append(sess['masks'][fi].copy())
    name = 'mask_{:03d}'.format(len(sess['multi_mask']['masks']))
    sess['multi_mask']['mask_names'].append(name)
    sess['click_state'] = [[], []]

    # Repaint with all masks
    frame = sess['origin_images'][fi].copy()
    for i, m in enumerate(sess['multi_mask']['masks']):
        frame = mask_painter(frame, m.astype('uint8'), mask_color=i + 2)
    painted = Image.fromarray(frame)
    sess['painted_images'][fi] = painted

    return {'frame': frame_to_b64(painted), 'mask_names': sess['multi_mask']['mask_names']}


@app.post("/clear-clicks")
def clear_clicks(req: SessionReq):
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found")
    sess = sessions[req.session_id]
    sess['click_state'] = [[], []]
    fi = sess['select_frame_number']
    frame = sess['origin_images'][fi]
    return {'frame': frame_to_b64(frame)}


@app.post("/remove-masks")
def remove_masks(req: SessionReq):
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found")
    sess = sessions[req.session_id]
    sess['multi_mask'] = {'mask_names': [], 'masks': []}
    sess['click_state'] = [[], []]
    fi = sess['select_frame_number']
    return {'frame': frame_to_b64(sess['origin_images'][fi]), 'mask_names': []}


@app.post("/select-frame")
def select_frame(req: SelectFrameReq):
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found")
    sess = sessions[req.session_id]
    idx = max(0, min(req.frame_index, len(sess['origin_images']) - 1))
    sess['select_frame_number'] = idx
    sess['click_state'] = [[], []]
    sam = get_sam()
    sam.sam_controler.reset_image()
    sam.sam_controler.set_image(sess['origin_images'][idx])
    return {'frame': frame_to_b64(sess['painted_images'][idx])}


def do_matting(req: RunReq, jid: str):
    """Run matting for one session. Updates jobs[jid] in place."""
    try:
        sid = req.session_id
        sess = sessions[sid]
        fi = sess['select_frame_number']
        all_frames = sess['origin_images']   # ALL frames — wrapper handles bidirectional split

        tmpl = build_template_mask(sess, req.selected_masks)
        if len(np.unique(tmpl)) == 1:
            tmpl[0][0] = 1

        jobs[jid]['progress'] = 15
        model = get_matanyone()
        processor = InferenceCore(model, cfg=model.cfg)
        processor.memory.max_mem_frames = max(1, req.max_mem_frames - 1)
        jobs[jid]['progress'] = 25
        jobs[jid]['phase'] = 'warmup'
        jobs[jid]['frame'] = 0
        jobs[jid]['total'] = 0

        def on_progress(current, total, phase):
            # backward and forward together fill 35→85%
            pct = 35 + int((current / max(total, 1)) * 50)
            jobs[jid].update({'progress': pct, 'phase': phase, 'frame': current, 'total': total})

        vname = os.path.splitext(sess['video_name'])[0]
        fg_temp = os.path.join(RESULTS_DIR, f"{sid}_{vname}_fg_temp.mp4")
        al_path = os.path.join(RESULTS_DIR, f"{sid}_{vname}_alpha.mp4")

        matanyone2_fn(
            processor, all_frames, tmpl * 255,
            r_erode=req.erode, r_dilate=req.dilate,
            progress_callback=on_progress,
            fg_path=fg_temp, alpha_path=al_path, fps=sess['fps'],
            start_frame=fi,
        )
        jobs[jid]['progress'] = 85

        fg_final = os.path.join(RESULTS_DIR, f"{sid}_{vname}_fg.mp4")
        if sess.get('audio') and os.path.exists(sess['audio']):
            fg_out = add_audio_to_video(fg_temp, sess['audio'], fg_final)
            if fg_out == fg_final and os.path.exists(fg_temp) and fg_temp != fg_final:
                try:
                    os.remove(fg_temp)
                except Exception:
                    pass
        else:
            import shutil
            shutil.move(fg_temp, fg_final)
            fg_out = fg_final

        jobs[jid].update({
            'status': 'done', 'progress': 100,
            'fg': os.path.basename(fg_out),
            'alpha': os.path.basename(al_path),
        })
    except Exception as e:
        import traceback
        jobs[jid].update({'status': 'error', 'error': str(e), 'trace': traceback.format_exc()})


@app.post("/run")
def run_matting(req: RunReq):
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found")
    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'running', 'progress': 0}
    t = threading.Thread(target=do_matting, args=(req, job_id), daemon=True)
    t.start()
    return {'job_id': job_id}


@app.post("/batch-run")
def batch_run(req: BatchRunReq):
    for item in req.items:
        if item.session_id not in sessions:
            raise HTTPException(404, f"Session {item.session_id} not found")

    batch_id = str(uuid.uuid4())
    job_ids = []
    for item in req.items:
        jid = str(uuid.uuid4())
        jobs[jid] = {'status': 'pending', 'progress': 0,
                     'video_name': sessions[item.session_id]['video_name']}
        job_ids.append(jid)

    batches[batch_id] = {'status': 'running', 'job_ids': job_ids, 'current': 0}

    def run_all():
        for i, (item, jid) in enumerate(zip(req.items, job_ids)):
            batches[batch_id]['current'] = i
            jobs[jid]['status'] = 'running'
            do_matting(item, jid)
        batches[batch_id]['status'] = 'done'

    t = threading.Thread(target=run_all, daemon=True)
    t.start()
    return {'batch_id': batch_id, 'job_ids': job_ids}


@app.get("/batch-status/{batch_id}")
def batch_status(batch_id: str):
    if batch_id not in batches:
        raise HTTPException(404, "Batch not found")
    b = batches[batch_id]
    return {**b, 'jobs': {jid: jobs.get(jid, {}) for jid in b['job_ids']}}


@app.get("/session-info/{session_id}")
def session_info(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    sess = sessions[session_id]
    fi = sess['select_frame_number']
    return {
        'total_frames': len(sess['origin_images']),
        'fps': sess['fps'],
        'image_w': sess['origin_images'][0].shape[1],
        'image_h': sess['origin_images'][0].shape[0],
        'mask_names': sess['multi_mask']['mask_names'],
        'current_frame': fi,
        'frame': frame_to_b64(sess['painted_images'][fi]),
    }


@app.get("/status/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/result/{filename}")
def get_result(filename: str):
    if '..' in filename or '/' in filename:
        raise HTTPException(400, "Invalid filename")
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type='video/mp4',
                        headers={'Content-Disposition': f'attachment; filename="{filename}"'})


if __name__ == '__main__':
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    a = parser.parse_args()
    uvicorn.run(app, host='0.0.0.0', port=a.port)
