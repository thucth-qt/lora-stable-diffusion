#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="/data/raw_models/dnd/raw_weights/pretrained_pipes/realistic_V5.1"
export INSTANCE_DIR="/home/thucth/thucth_dev/SD/lora/Van_cropped"
export OUTPUT_DIR="/home/thucth/thucth_dev/SD/lora/training_output_Van"
export INSTNACE_PROMPT="van"


CUDA_VISIBLE_DEVICES="2" python3 train_lora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt=$INSTNACE_PROMPT \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --learning_rate_text=5e-5 \
  --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10000 