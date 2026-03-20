
import gc
import tqdm
import torch
import imageio
from torchvision.transforms.functional import to_tensor
import numpy as np
import random
import cv2
from matanyone2.utils.device import get_default_device, safe_autocast_decorator, clean_vram

device = get_default_device()

def gen_dilate(alpha, min_kernel_size, max_kernel_size):
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1) * 255
    return dilate.astype(np.float32)

def gen_erosion(alpha, min_kernel_size, max_kernel_size):
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    erode = cv2.erode(fg, kernel, iterations=1) * 255
    return erode.astype(np.float32)


def _mps_flush():
    gc.collect()
    if hasattr(torch, 'mps'):
        try:
            torch.mps.synchronize()
            torch.mps.empty_cache()
        except Exception:
            pass


def _cleanup_processor(proc):
    proc.clear_memory()
    proc.last_mask = None
    proc.last_pix_feat = None
    proc.last_msk_value = None
    proc.image_feature_store._store.clear()
    clean_vram()


def _run_pass_collect(proc, frames, mask_tensor, n_warmup, bgr, objects, progress_callback, phase):
    """Run one inference pass and collect results in memory as uint8 numpy arrays.

    Returns:
        fg_frames: list of (H, W, 3) uint8
        alpha_frames: list of (H, W, 1) uint8  — one per *output* frame (warmup excluded)
    """
    augmented = [frames[0]] * n_warmup + list(frames)
    total = len(augmented)
    fg_out, al_out = [], []
    local_mask = mask_tensor

    for ti, frame in tqdm.tqdm(enumerate(augmented), total=total, desc=phase):
        image = to_tensor(frame).float().to(device)
        if ti == 0:
            prob = proc.step(image, local_mask, objects=objects)
            prob = proc.step(image, first_frame_pred=True)
        elif ti <= n_warmup:
            prob = proc.step(image, first_frame_pred=True)
        else:
            prob = proc.step(image)
        del image
        local_mask = proc.output_prob_to_mask(prob)
        del prob
        pha = local_mask.unsqueeze(2).detach().to('cpu').numpy()
        if ti >= n_warmup:
            com = frame / 255. * pha + bgr * (1 - pha)
            fg_out.append((com * 255).astype(np.uint8))
            al_out.append((pha * 255).astype(np.uint8))
        _mps_flush()
        if progress_callback is not None:
            if ti < n_warmup:
                progress_callback(ti + 1, n_warmup, 'warmup_' + phase)
            else:
                progress_callback(ti - n_warmup + 1, len(frames), phase)

    _cleanup_processor(proc)
    return fg_out, al_out


@torch.inference_mode()
@safe_autocast_decorator()
def matanyone2(processor, frames_np, mask, r_erode=0, r_dilate=0, n_warmup=10,
               progress_callback=None, fg_path=None, alpha_path=None, fps=30,
               start_frame=0):
    """
    Bidirectional video matting.

    Args:
        processor:    InferenceCore instance (forward pass)
        frames_np:    ALL frames of the video as list of (H,W,3) uint8
        mask:         (H,W) uint8 annotation mask at start_frame
        start_frame:  index of the annotated frame in frames_np (default 0 = forward only)
        fg_path / alpha_path: stream output to disk if provided

    Outputs:
        (fg_path, alpha_path) if paths given, else (fg_list, alpha_list)
    """
    bgr = (np.array([120, 255, 155], dtype=np.float32) / 255).reshape((1, 1, 3))
    objects = [1]

    if r_dilate > 0:
        mask = gen_dilate(mask, r_dilate, r_dilate)
    if r_erode > 0:
        mask = gen_erosion(mask, r_erode, r_erode)

    mask_tensor = torch.from_numpy(mask).to(device)

    # ── Backward pass ────────────────────────────────────────────────────────
    back_fg, back_alpha = [], []
    if start_frame > 0:
        from matanyone2.inference.inference_core import InferenceCore
        back_proc = InferenceCore(processor.network, processor.cfg)

        # frames in reverse order: [fi, fi-1, ..., 0]
        backward_frames = list(reversed(frames_np[:start_frame + 1]))

        def back_progress(cur, tot, phase):
            # Report backward progress as first half of total work
            total_output = start_frame + (len(frames_np) - start_frame)
            if progress_callback:
                progress_callback(cur, total_output, 'backward')

        bfg, bal = _run_pass_collect(
            back_proc, backward_frames, mask_tensor.clone(),
            n_warmup, bgr, objects, back_progress, 'backward'
        )
        # bfg = [result_fi, result_fi-1, ..., result_0]
        # Reverse and drop result_fi (covered by forward's first output frame)
        back_fg = list(reversed(bfg[1:]))    # [result_0, ..., result_fi-1]
        back_alpha = list(reversed(bal[1:]))
        del bfg, bal

    # ── Forward pass ─────────────────────────────────────────────────────────
    forward_frames = frames_np[start_frame:]          # [fi, fi+1, ..., end]
    fwd_augmented = [forward_frames[0]] * n_warmup + list(forward_frames)
    total_output = len(back_fg) + len(forward_frames)

    fg_writer = alpha_writer = None
    frames_out, phas_out = [], []

    if fg_path and alpha_path:
        fg_writer = imageio.get_writer(
            fg_path, fps=fps, quality=7, codec='libx264', pixelformat='yuv420p')
        alpha_writer = imageio.get_writer(
            alpha_path, fps=fps, quality=7, codec='libx264', pixelformat='yuv420p')

    try:
        # Write backward frames first (already computed, stored in RAM)
        for fg_f, al_f in zip(back_fg, back_alpha):
            if fg_writer is not None:
                fg_writer.append_data(fg_f)
                alpha_writer.append_data(np.repeat(al_f, 3, axis=2))
            else:
                frames_out.append(fg_f)
                phas_out.append(al_f)

        # Run forward pass inline (streaming)
        fwd_mask = mask_tensor
        for ti, frame_single in tqdm.tqdm(
                enumerate(fwd_augmented), total=len(fwd_augmented), desc='forward'):
            image = to_tensor(frame_single).float().to(device)
            if ti == 0:
                output_prob = processor.step(image, fwd_mask, objects=objects)
                output_prob = processor.step(image, first_frame_pred=True)
            elif ti <= n_warmup:
                output_prob = processor.step(image, first_frame_pred=True)
            else:
                output_prob = processor.step(image)
            del image
            fwd_mask = processor.output_prob_to_mask(output_prob)
            del output_prob
            pha = fwd_mask.unsqueeze(2).detach().to('cpu').numpy()
            com_np = frame_single / 255. * pha + bgr * (1 - pha)
            if ti >= n_warmup:
                fg_frame = (com_np * 255).astype(np.uint8)
                alpha_frame = np.repeat((pha * 255).astype(np.uint8), 3, axis=2)
                if fg_writer is not None:
                    fg_writer.append_data(fg_frame)
                    alpha_writer.append_data(alpha_frame)
                else:
                    frames_out.append(fg_frame)
                    phas_out.append((pha * 255).astype(np.uint8))
            _mps_flush()
            if progress_callback is not None:
                fwd_out_idx = (ti - n_warmup + 1) if ti >= n_warmup else 0
                done = len(back_fg) + fwd_out_idx
                phase = 'warmup' if ti < n_warmup else 'inference'
                progress_callback(done, total_output, phase)
    finally:
        if fg_writer is not None:
            fg_writer.close()
        if alpha_writer is not None:
            alpha_writer.close()

    _cleanup_processor(processor)

    if fg_path and alpha_path:
        return fg_path, alpha_path
    return frames_out, phas_out
