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
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightning", "peft", "diffusers", "huggingface-hub", "pillow", "imageio", "pandas", "einops"])
    
    # Install diffsynth from current repo in editable mode
    current_dir = os.getcwd()
    if os.path.exists(os.path.join(current_dir, "setup.py")) or os.path.exists(os.path.join(current_dir, "diffsynth")):
        print("Installing diffsynth from local repo...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", "."])
    
    print("✓ Dependencies installed!")

# Install dependencies FIRST
setup_dependencies()

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

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
        
        # Import here to avoid module not found errors
        try:
            from diffsynth import ModelManager, WanVideoMoChaPipeline
        except ImportError as e:
            print(f"Error importing diffsynth: {e}")
            print("Make sure you've cloned the MoCha repo and are in the correct directory")
            raise
        
        from peft import LoraConfig, inject_adapter_in_model
        
        device = torch.device("cpu")
        
        model_manager = ModelManager(torch_dtype=torch.float32, device=device)
        
        # Load model paths
        if self.use_1_3b:
            print("Loading Wan2.1-T2V-1.3B...")
            model_manager.load_models([
                "Wan-AI/Wan2.1-T2V-1.3B",
                "./models/models_t5_umt5-xxl-enc-bf16.pth",
                "./models/Wan2.1_VAE.pth",
            ])
        else:
            print("Loading Wan2.1-T2V-14B...")
            model_manager.load_models([
                "./models/diffusion_pytorch_model.safetensors",
                "./models/models_t5_umt5-xxl-enc-bf16.pth",
                "./models/Wan2.1_VAE.pth",
            ])
        
        self.pipe = WanVideoMoChaPipeline.from_model_manager(model_manager, device=device)
        
        # Freeze base model
        self.pipe.requires_grad_(False)
        
        # Inject LoRA
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=["q", "k", "v", "o"],
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        
        print("Injecting LoRA adapters...")
        self.pipe.dit = inject_adapter_in_model(lora_config, self.pipe.dit)
        self.pipe.train()
        print("✓ Model loaded with LoRA!")

    def training_step(self, batch, batch_idx):
        device = self.device
        
        # ===== LATENT CACHING =====
        if batch["needs_cache"][0]:
            video = batch["video"].to(device)
            print(f"[Batch {batch_idx}] Caching latent...")
            with torch.no_grad():
                latents = self.pipe.vae.encode(video, device=device)
            torch.save(latents.cpu(), batch["latent_path"][0])
        else:
            latents = torch.load(batch["latent_path"][0]).to(device)
        
        # ===== DIFFUSION PROCESS =====
        noise = torch.randn_like(latents)
        
        t_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,), device=device)
        timestep = self.pipe.scheduler.timesteps[t_id]
        
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        
        # ===== FORWARD PASS =====
        with torch.no_grad():
            prompt_emb = self.pipe.encode_prompt("")
        
        noise_pred = self.pipe.dit(
            noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_emb["context"],
        )
        
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
    
    # Check if we're in the right directory
    if not os.path.exists("diffsynth"):
        print("❌ ERROR: diffsynth folder not found!")
        print("\nFor Google Colab, use these commands:")
        print("  !git clone https://github.com/darurat14/mocha.git")
        print("  %cd mocha")
        print("  !python train_mocha_colab.py --use_1_3b")
        sys.exit(1)
    
    if not os.path.exists("./data/train_data.csv"):
        print("⚠️  WARNING: ./data/train_data.csv not found!")
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
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    
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
    
    # Setup trainer
    device = "gpu" if (args.use_gpu and torch.cuda.is_available()) else "cpu"
    print(f"Training on: {device.upper()}")
    
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
        accelerator=device,
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        enable_progress_bar=True,
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
