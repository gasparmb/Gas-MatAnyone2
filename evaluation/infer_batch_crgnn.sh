#!/bin/bash

input_folder="data/crgnn/image_allframe"
mask_folder="data/crgnn/mask"

ckpt_name="matanyone2"

for video_folder in "${input_folder}"/*; do
  if [ -d "${video_folder}" ]; then
    video_id=$(basename "${video_folder}")

    mask_file="${mask_folder}/${video_id}.png"
    if [ -f "${mask_file}" ]; then

      input_frames_folder="${video_folder}"
      if [ -d "${input_frames_folder}" ]; then
        echo "Processing video: ${video_id}"
        
        python inference_matanyone2.py \
                --input_path "${input_frames_folder}" \
                --mask_path "${mask_file}" \
                --output_path "data/results/crgnn" \
                --ckpt_path "pretrained_models/${ckpt_name}.pth" \
                --warmup 10 \
                --erode_kernel 10 \
                --dilate_kernel 10 \
                --save_image
      fi
    fi
  fi
done
