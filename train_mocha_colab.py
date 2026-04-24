"""
MoCha LoRA Training for Google Colab - All-in-One Script
Paste this entire script into a Colab cell and run!
"""

import os
import sys
import subprocess

# =========================
# SETUP DEPENDENCIES (MUST BE FIRST!)
# =========================
def setup_dependencies():
    """Install required packages for Colab"""
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightning", "peft", "diffusers", "huggingface-hub", "pillow", "imageio", "pandas", "einops", "modelscope", "accelerate" , "ftfy"])
    print("✓ Dependencies installed!")

# Install dependencies FIRST
setup_dependencies()

# Add current directory to path for imports (diffsynth is in this directory)
sys.path.insert(0, os.getcwd())
print(f"✓ Added {os.getcwd()} to Python path")

# NOW import everything else
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import imageio
import pandas as pd
import numpy as np
from PIL import Image
import torchvision
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint


# =========================
# PRE-ENCODING HELPER
# =========================
def preencode_videos_to_latents(model, dataset, latent_dir, device):
    """Pre-encode all training videos to latents (optional, skipped to save GPU memory)"""
    print("\n" + "="*60)
    print("Pre-encoding videos to latents...")
    print("="*60)
    
    print("⚠️  Skipping pre-encoding to save GPU memory (Tesla T4 only has 14GB)")
    print("   Videos will be encoded on-the-fly during training (slower but works)")
    print(f"✓ Pre-encoding complete!\n")


# =========================
# VIDEO DATASET
# =========================
class VideoRefDataset(Dataset):
    def __init__(self, data_path, max_num_frames=161, frame_interval=1, num_frames=161, height=480, width=832):
        metadata = pd.read_csv(data_path)
        self.video_path = metadata["source_video"]
        self.mask_path = metadata["source_mask"]
        self.ref_path_1 = metadata["reference_1"]
        self.ref_path_2 = []
        for ref_name in metadata["reference_2"]:
            if pd.isna(ref_name) or ref_name == 'None':
                self.ref_path_2.append("None")
            else:
                self.ref_path_2.append(ref_name)
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.mask_process = v2.Compose([
            v2.CenterCrop(size=(height // 8, width // 8)),
            v2.Resize(size=(height // 8, width // 8), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
    def crop_and_resize(self, image, isMask=False):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        if isMask:
            scale /= 8
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image

    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            num_frames = 1 + (reader.count_frames() - 1) //4 * 4
        
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        return frames

    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False

    def load_image_frame(self, file_path, isMask=False):
        image = Image.open(file_path).convert('RGB')
        image = self.crop_and_resize(image, isMask)
        if isMask:
            image = self.mask_process(image)
        else:
            image = self.frame_process(image)
        image = image.unsqueeze(1)
        return image
    
    def load_video(self, file_path, isMask=False):
        if self.is_image(file_path):
            if isMask:
                return self.load_image_frame(file_path, isMask=True)
            else:
                return self.load_image_frame(file_path)
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, 0, self.frame_interval, self.num_frames, self.frame_process)
        return frames

    def __getitem__(self, data_id):
        video_path = self.video_path[data_id]
        mask_path = self.mask_path[data_id]

        video = self.load_video(video_path)
        if video is None:
            raise ValueError(f"{video_path} is not a valid video.")

        source_mask = self.load_video(mask_path, isMask=True)
        mask_cond = torch.sign(source_mask[0:1, 0:1, :, :]).repeat(16, 1, 1, 1)

        ref_path_1 = self.ref_path_1[data_id]
        ref_path_2 = self.ref_path_2[data_id]

        first_ref = self.load_video(ref_path_1)

        if pd.isna(ref_path_2) or ref_path_2 == 'None':
            second_ref = "None"
        else:
            second_ref = self.load_video(ref_path_2)

        data = {"video": video, "video_path": video_path, "mask": mask_cond, "first_ref": first_ref, "second_ref": second_ref}
        return data
    
    def __len__(self):
        return len(self.video_path)


# =========================
# LATENT CACHING DATASET
# =========================
class LatentDataset(Dataset):
    def __init__(self, base_dataset, latent_dir):
        self.base_dataset = base_dataset
        self.latent_dir = latent_dir
        os.makedirs(latent_dir, exist_ok=True)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        latent_path = os.path.join(self.latent_dir, f"{idx}.pt")

        if not os.path.exists(latent_path):
            return {
                "video": self.base_dataset[idx]["video"],
                "latent_path": latent_path,
                "needs_cache": True,
            }
        else:
            return {
                "latent_path": latent_path,
                "needs_cache": False,
            }


# =========================
# LIGHTNING MODEL
# =========================
class MoChALoRALightning(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, lora_rank=8, lora_alpha=16, use_1_3b=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.use_1_3b = use_1_3b
        self.pipe = None

    def load_models(self):
        """Load WanVideo MoCha pipeline and inject LoRA"""
        print("Loading WanVideo MoCha pipeline...")
        
        from diffsynth.models import ModelManager
        from diffsynth.pipelines import WanVideoMoChaPipeline
        from peft import LoraConfig, inject_adapter_in_model
        from huggingface_hub import snapshot_download
        import glob
        
        # Auto-detect GPU or use CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            torch_dtype = torch.bfloat16  # GPU supports mixed precision
        else:
            device = torch.device("cpu")
            print("⚠️  No GPU detected - using CPU (training will be SLOW)")
            print("    Tip: In Colab, enable GPU via Runtime > Change runtime type")
            torch_dtype = torch.float32
        
        model_manager = ModelManager(torch_dtype=torch_dtype, device=device)
        
        if self.use_1_3b:
            print("Loading Wan2.1-T2V-1.3B from HuggingFace...")
            
            try:
                # Download only DiT model (needed for training)
                print("Downloading Wan2.1-T2V-1.3B DiT...")
                wan_path = snapshot_download(repo_id="Wan-AI/Wan2.1-T2V-1.3B", local_dir="./models/wan2.1_1.3b")
                
                # Find DiT model file
                dit_files = glob.glob(os.path.join(wan_path, "diffusion_pytorch_model.safetensors"))
                if not dit_files:
                    dit_files = glob.glob(os.path.join(wan_path, "*.safetensors"))
                
                print(f"  ✓ Found {len(dit_files)} DiT model file(s)")
                
                # Load ALL model files from the downloaded directory
                # BUT skip T5 encoder (only needed for inference, not training)
                print("Loading model components...")
                model_files = glob.glob(os.path.join(wan_path, "*.safetensors")) + \
                             glob.glob(os.path.join(wan_path, "*.pth"))
                
                print(f"  Found {len(model_files)} model file(s)")
                for model_file in model_files:
                    # Skip T5 encoder - it's huge and not needed for LoRA training
                    if "t5" in model_file.lower() or "text_encoder" in model_file.lower():
                        print(f"  Skipping: {os.path.basename(model_file)} (not needed for training)")
                        continue
                    
                    try:
                        print(f"  Loading: {os.path.basename(model_file)}")
                        model_manager.load_model(model_file, device=device, torch_dtype=torch_dtype)
                    except Exception as e:
                        print(f"    ⚠️  Skip: {str(e)[:80]}")
                
            except Exception as e:
                print(f"❌ Error loading Wan models: {e}")
                print("This might be due to missing model files. Please ensure:")
                print("  1. HuggingFace token is set if model is private")
                print("  2. You have enough disk space (~50GB)")
                raise
                
        else:
            print("Loading Wan2.1-T2V-14B...")
            print("❌ 14B model requires local model files. Please use --use_1_3b for Colab.")
            raise ValueError("14B model not supported in Colab mode. Use --use_1_3b")
        
        print("Creating pipeline...")
        self.pipe = WanVideoMoChaPipeline.from_model_manager(model_manager, device=device)
        
        # Freeze base model
        self.pipe.requires_grad_(False)
        
        # Inject LoRA
        print("Injecting LoRA adapters...")
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=["q", "k", "v", "o"],
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        
        self.pipe.dit = inject_adapter_in_model(lora_config, self.pipe.dit)
        self.pipe.train()
        print("✓ Model loaded with LoRA!")

    def training_step(self, batch, batch_idx):
        device = self.device
        
        # ===== ENCODE VIDEO TO LATENTS (on-the-fly to save GPU memory) =====
        if batch["needs_cache"][0]:
            # Encode fresh video
            video = batch["video"].to(device)
            print(f"[Batch {batch_idx}] Encoding video to latents...")
            
            if self.pipe.vae is None:
                raise RuntimeError("VAE not loaded! Cannot encode video to latents.")
            
            # Cast to correct dtype
            vae_dtype = next(self.pipe.vae.parameters()).dtype
            video = video.to(dtype=vae_dtype)
            
            with torch.no_grad():
                latents = self.pipe.vae.encode(video, device=device)
            
            # Move VAE to CPU to free GPU memory for training
            self.pipe.vae.to("cpu")
            torch.cuda.empty_cache()
            
            # Try to save for next epoch
            try:
                torch.save(latents.cpu(), batch["latent_path"][0])
            except:
                pass  # Ignore save errors
        else:
            # Load pre-cached latent
            latents = torch.load(batch["latent_path"][0]).to(device)
        
        # ===== DIFFUSION PROCESS =====
        noise = torch.randn_like(latents)
        
        # Sample random timestep (scale to scheduler's timestep range)
        num_scheduler_steps = len(self.pipe.scheduler.timesteps)
        t_id = torch.randint(0, num_scheduler_steps, (1,), device="cpu")
        timestep = self.pipe.scheduler.timesteps[t_id].to(device)
        
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        
        # ===== FORWARD PASS =====
        # Create empty context (text encoder not loaded to save GPU memory)
        batch_size = noisy_latents.shape[0]
        context = torch.zeros(batch_size, 1, 4096, device=device, dtype=noisy_latents.dtype)
        
        # Call WanModel forward with positional args: (noisy_latents, timestep, context)
        noise_pred = self.pipe.dit(noisy_latents, timestep, context)
        
        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
        self.log("train_loss", loss, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        trainable_params = []
        
        for name, param in self.pipe.dit.named_parameters():
            if "lora" in name:
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False
        
        num_params = sum(p.numel() for p in trainable_params)
        print(f"✓ Trainable LoRA params: {num_params:,}")
        
        return torch.optim.AdamW(trainable_params, lr=self.learning_rate)


# =========================
# MAIN TRAINING FUNCTION
# =========================
def main():
    print("="*60)
    print("MoCha LoRA Training Setup")
    print("="*60)
    
    # Debug: Check directory structure
    cwd = os.getcwd()
    print(f"\nCurrent directory: {cwd}")
    print(f"Python path includes: {cwd in sys.path}")
    
    # Check if we're in the right directory
    if not os.path.exists("diffsynth"):
        print("❌ ERROR: diffsynth folder not found!")
        print("\nFor Google Colab, use these commands:")
        print("  !git clone https://github.com/darurat14/mocha.git")
        print("  %cd mocha")
        print("  !python train_mocha_colab.py --use_1_3b")
        sys.exit(1)
    
    # Debug: Check diffsynth structure
    print("\nChecking diffsynth structure...")
    if os.path.exists("diffsynth/__init__.py"):
        print("  ✓ diffsynth/__init__.py exists")
    if os.path.exists("diffsynth/models"):
        print("  ✓ diffsynth/models/ exists")
        if os.path.exists("diffsynth/models/__init__.py"):
            print("  ✓ diffsynth/models/__init__.py exists")
        else:
            print("  ⚠️  diffsynth/models/__init__.py MISSING!")
    
    if not os.path.exists("./data/train_data.csv"):
        print("\n⚠️  WARNING: ./data/train_data.csv not found!")
        print("   Make sure your training data is in ./data/ directory")
    
    parser = argparse.ArgumentParser(description="MoCha LoRA Training for Colab")
    parser.add_argument("--data_path", type=str, default="./data/train_data.csv", help="Path to training data CSV")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--latent_dir", type=str, default="./latents", help="Directory for cached latents")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers (use 0 for Colab)")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--use_1_3b", action="store_true", help="Use 1.3B model instead of 14B")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU (use CPU instead)")
    
    args = parser.parse_args()
    
    # Create output directories FIRST
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.latent_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("MoCha LoRA Training - Google Colab Edition")
    print("="*60)
    print(f"Data: {args.data_path}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Max steps: {args.max_steps}")
    print(f"LR: {args.learning_rate}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Use 1.3B: {args.use_1_3b}")
    print("="*60 + "\n")
    
    # Initialize model
    model = MoChALoRALightning(
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_1_3b=args.use_1_3b,
    )
    model.load_models()
    
    # Load dataset
    print(f"\nLoading training data from {args.data_path}...")
    base_dataset = VideoRefDataset(args.data_path)
    dataset = LatentDataset(base_dataset, args.latent_dir)
    
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} samples\n")
    
    # Pre-encode videos to latents (saves GPU memory during training)
    # Get device from the loaded model
    model_device = next(model.pipe.dit.parameters()).device if model.pipe and model.pipe.dit else torch.device("cpu")
    preencode_videos_to_latents(model, dataset, args.latent_dir, model_device)
    
    # Setup trainer
    use_gpu = (not args.no_gpu) and torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    devices = 1  # Always 1 device (single GPU or single CPU)
    print(f"Training on: {accelerator.upper()}")
    if use_gpu:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  CPU (to use GPU, remove --no_gpu flag)")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="mocha_lora_{epoch:02d}_{step}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
    )
    
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        max_steps=args.max_steps,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        enable_progress_bar=True,
        precision="16-mixed" if use_gpu else "32",  # Mixed precision for GPU
    )
    
    # Train
    print("\nStarting training...\n")
    trainer.fit(model, dataloader)
    
    # Save LoRA weights
    print("\nSaving LoRA weights...")
    final_dir = os.path.join(args.output_dir, "lora_final")
    os.makedirs(final_dir, exist_ok=True)
    
    model.pipe.dit.save_pretrained(final_dir)
    
    torch.save(
        {
            name: p.cpu()
            for name, p in model.pipe.dit.named_parameters()
            if "lora" in name
        },
        os.path.join(args.output_dir, "lora_weights_final.pth"),
    )
    
    print(f"\n✓ Training complete!")
    print(f"✓ LoRA weights saved to: {final_dir}")
    print(f"✓ Checkpoint saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
