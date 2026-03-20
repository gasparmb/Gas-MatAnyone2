#!/bin/bash

input_folder="./data/VideoMatte/videomatte_1920x1080"
mask_folder="./data/VideoMatte_first_frame_seg_mask/videomatte_1920x1080"

ckpt_name="matanyone2"

for subfolder in "videomatte_motion" "videomatte_static"; do
  subfolder_path="${input_folder}/${subfolder}"
  
  echo "Processing subfolder: ${subfolder}"
  
  for video_folder in "${subfolder_path}"/*; do
    if [ -d "${video_folder}" ]; then
      video_id=$(basename "${video_folder}")

      mask_file="${mask_folder}/${subfolder}/${video_id}.png"
      if [ -f "${mask_file}" ]; then

        input_frames_folder="${video_folder}/com"
        if [ -d "${input_frames_folder}" ]; then
          echo "Processing video: ${video_id} from ${subfolder}"
          
          python evaluation/inference_matanyone_eval.py \
                  --input_path "${input_frames_folder}" \
                  --mask_path "${mask_file}" \
                  --output_path "./data/results/videomatte_1920x1080/${subfolder}" \
                  --ckpt_path "pretrained_models/${ckpt_name}.pth" \
                  --warmup 10 \
                  --erode_kernel 15 \
                  --dilate_kernel 15 \
                  --save_image
        fi
      fi
    fi
  done
done

