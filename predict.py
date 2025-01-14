# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path, Secret
import os
import time
import shutil
import subprocess
from zipfile import ZipFile, is_zipfile

GPU_IDS = "0"
DATA_ROOT = "dataset"
OUTPUT_DIR= "output"
MODEL_CACHE = "HunyuanVideo"
MODEL_URL = "https://weights.replicate.delivery/default/hunyuanvideo-community/HunyuanVideo/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Set environment variables
        os.environ["WANDB_MODE"] = "offline"
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
        os.environ["FINETRAINERS_LOG_LEVEL"] = "INFO"
        
        # Download weights
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

    def predict(
        self,
        input_videos: Path = Input(description="ZIP file containing video dataset"),
        trigger_word: str = Input(description="Trigger word", default="afkx"),
        train_steps: int = Input(description="Number of training steps", default=500, ge=10, le=4000),
        rank: int = Input(description="LoRA rank", default=128, ge=16, le=128),
        video_resolution_buckets: str = Input(description="--video_resolution_buckets 1x512x768 33x512x768", default="49x512x768"),
        batch_size: int = Input(description="Batch size", default=1, ge=1, le=4),
        gradient_accumulation_steps: int = Input(description="Gradient accumulation steps", default=1),
        seed: int = Input(description="Random seed", default=0),
        hub_model_id: str = Input(description="Hugging Face model path to upload trained LoRA", default=None),
        hf_token: Secret = Input(description="Hugging Face token for model upload", default=None),
    ) -> Path:
        """Run training pipeline"""
        if seed <=0:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # Cleanup past runs
        print("Cleaning up past runs")
        if os.path.exists(DATA_ROOT):
            shutil.rmtree(DATA_ROOT)
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)

        # Check if input_videos is a zip file
        if not is_zipfile(input_videos):
            raise ValueError("input_images must be a zip file")

        # Extract files from the zip file
        os.makedirs(DATA_ROOT, exist_ok=True)
        file_count = 0
        with ZipFile(input_videos, "r") as zip_ref:
            for file_info in zip_ref.infolist():
                if not file_info.filename.startswith(
                    "__MACOSX/"
                ) and not file_info.filename.startswith("._"):
                    zip_ref.extract(file_info, DATA_ROOT)
                    file_count += 1

        print(f"Extracted {file_count} files from zip to folder: {DATA_ROOT}")

        # Set training arguments
        training_args = [
            "accelerate",
            "launch",
            "--config_file", "accelerate_configs/compiled_1.yaml",
            "--gpu_ids", GPU_IDS,
            "train.py",
            "--model_name", "hunyuan_video",
            "--pretrained_model_name_or_path", MODEL_CACHE,
            "--enable_tiling",
            "--enable_slicing",
            "--data_root", DATA_ROOT,
            "--caption_column", "prompts.txt",
            "--video_column", "videos.txt",
            "--seed", str(seed),
            "--rank", str(rank),
            "--lora_alpha", str(rank),
            "--mixed_precision", "bf16",
            "--output_dir", OUTPUT_DIR,
            "--batch_size", str(batch_size),
            "--id_token", trigger_word,
            "--video_resolution_buckets", video_resolution_buckets,
            "--caption_dropout_p", str(0.05),
            "--training_type", "lora",
            "--train_steps", str(train_steps),
            "--gradient_accumulation_steps", str(gradient_accumulation_steps),
            "--gradient_checkpointing",
            "--precompute_conditions",
            "--use_8bit_bnb",
            "--optimizer", "adamw",
            "--lr", str(2e-5),
            "--lr_scheduler", "constant_with_warmup",
            "--lr_warmup_steps", str(100),
            "--lr_num_cycles", str(1),
            "--beta1", str(0.9),
            "--beta2", str(0.95),
            "--max_grad_norm", str(1.0),
            "--weight_decay", str(1e-4),
            "--epsilon", str(1e-8),
            "--tracker_name", "replicate-hunyuanvideo"
        ]

        # Run the trainer
        print(f"Using args: {training_args}")
        subprocess.run(training_args, check=True, close_fds=False)

        # Check to upload to Hugging Face
        if hf_token is not None and hub_model_id is not None:
            token = hf_token.get_secret_value()
            os.system(f"huggingface-cli login --token {token}")
            os.system(f"huggingface-cli upload {hub_model_id} {OUTPUT_DIR}")
        
        # Create output tar of trained model
        output_path = "/tmp/trained_model.tar"
        os.system(f"tar -cvf {output_path} -C {OUTPUT_DIR} .")
        return Path(output_path)
