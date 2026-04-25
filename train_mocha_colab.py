import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import argparse
import subprocess

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import imageio
import pandas as pd
from PIL import Image
import torchvision
from torchvision.transforms import v2
from einops import rearrange

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint


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
            item = self.base_dataset[idx]
            return {
                "video": item["video"],
                "latent_path": latent_path,
                "needs_cache": True,
            }

        return {
            "latent_path": latent_path,
            "needs_cache": False,
        }


class VideoRefDataset(Dataset):
    def __init__(self, data_path, num_frames=161, height=480, width=832):
        metadata = pd.read_csv(data_path)
        self.video_paths = metadata["source_video"]
        self.num_frames = num_frames
        self.height = height
        self.width = width

        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])

    def __len__(self):
        return len(self.video_paths)

    def is_image(self, path):
        return path.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))

    def load_image(self, path):
        img = Image.open(path).convert("RGB")
        img = self.frame_process(img)
        img = img.unsqueeze(1)
        return img

    def load_video(self, path):
        if self.is_image(path):
            return self.load_image(path)

        reader = imageio.get_reader(path)
        frames = []
        max_frames = min(self.num_frames, reader.count_frames())

        for i in range(max_frames):
            frame = Image.fromarray(reader.get_data(i))
            frame = self.frame_process(frame)
            frames.append(frame)

        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "t c h w -> c t h w")
        return frames

    def __getitem__(self, idx):
        video = self.load_video(self.video_paths[idx])
        return {"video": video}


class MoChALoRALightning(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, lora_rank=8, lora_alpha=16):
        super().__init__()
        self.learning_rate = learning_rate
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        self.dit = None
        self.vae = None
        self.scheduler = None
        self.runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_models(self):
        print("Loading WanVideo MoCha pipeline...")

        from diffsynth.models import ModelManager
        from diffsynth.pipelines import WanVideoMoChaPipeline
        from peft import LoraConfig, inject_adapter_in_model
        from huggingface_hub import snapshot_download
        import glob

        device = self.runtime_device
        torch_dtype = torch.float32

        if torch.cuda.is_available():
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ No GPU detected - using CPU")

        model_manager = ModelManager(
            torch_dtype=torch_dtype,
            device="cpu"
        )

        print("Loading Wan2.1-T2V-1.3B from HuggingFace...")
        try:
            wan_path = snapshot_download(
                repo_id="Wan-AI/Wan2.1-T2V-1.3B",
                local_dir="./models/wan2.1_1.3b"
            )

            model_files = glob.glob(os.path.join(wan_path, "*.safetensors")) + \
                         glob.glob(os.path.join(wan_path, "*.pth"))

            print(f"Found {len(model_files)} model file(s)")

            for model_file in model_files:
                if "t5" in model_file.lower() or "text_encoder" in model_file.lower():
                    print(f"Skipping: {os.path.basename(model_file)}")
                    continue

                try:
                    print(f"Loading: {os.path.basename(model_file)}")
                    model_manager.load_model(
                        model_file,
                        device="cpu",
                        torch_dtype=torch_dtype
                    )
                except Exception as e:
                    print(f"Skip: {str(e)[:80]}")

        except Exception as e:
            print(f"Error loading Wan models: {e}")
            raise

        pipe = WanVideoMoChaPipeline.from_model_manager(
            model_manager,
            device="cpu"
        )

        self.dit = pipe.dit
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler

        # keep VAE always on CPU
        if self.vae is not None:
            self.vae.to("cpu")
            self.vae.requires_grad_(False)

        # move ONLY DiT to GPU for training
        if torch.cuda.is_available():
            print("Moving DiT to GPU only...")
            self.dit = self.dit.to("cuda")
        else:
            print("Using CPU only for DiT")
            self.dit = self.dit.to("cpu")

        del pipe

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=["q", "k", "v", "o"],
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )

        self.dit = inject_adapter_in_model(lora_config, self.dit)

        # IMPORTANT FIX
        self.dit.to(device)
        self.dit.train()

        print("Model ready.")

    def training_step(self, batch, batch_idx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if batch["needs_cache"][0]:
            print(f"[Batch {batch_idx}] Encoding on CPU...")

            if self.vae is None:
                raise RuntimeError("VAE missing")

            video = batch["video"].cpu()
            vae_dtype = next(self.vae.parameters()).dtype
            video = video.to(dtype=vae_dtype)

            with torch.no_grad():
                latents = self.vae.encode(
                    video,
                    device="cpu"
                )

            # move only encoded latent to GPU
            latents = latents.to(device, non_blocking=True)
            torch.cuda.empty_cache()

            try:
                torch.save(latents.cpu(), batch["latent_path"][0])
            except Exception:
                pass

        else:
            latents = torch.load(batch["latent_path"][0]).to(device)

        noise = torch.randn_like(latents)

        num_steps = len(self.scheduler.timesteps)
        t_id = torch.randint(0, num_steps, (1,), device="cpu")
        timestep = self.scheduler.timesteps[t_id].to(device)

        noisy_latents = self.scheduler.add_noise(latents, noise, timestep)

        batch_size = noisy_latents.shape[0]

        # IMPORTANT FIX
        context = torch.zeros(
            batch_size,
            1,
            4096,
            device=device,
            dtype=noisy_latents.dtype,
        )

        noise_pred = self.dit(
            noisy_latents,
            timestep,
            context,
        )

        loss = torch.nn.functional.mse_loss(
            noise_pred.float(),
            noise.float(),
        )

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        trainable = []

        for name, param in self.dit.named_parameters():
            if "lora" in name:
                param.requires_grad = True
                trainable.append(param)
            else:
                param.requires_grad = False

        print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
        return torch.optim.AdamW(trainable, lr=self.learning_rate)


def main():
    parser = argparse.ArgumentParser(description="MoCha LoRA Training for Colab")
    parser.add_argument("--data_path", type=str, default="./data/train_data.csv", help="Path to training data CSV")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--latent_dir", type=str, default="./latents", help="Directory for cached latents")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")

    # compatibility args from original script
    parser.add_argument("--use_1_3b", action="store_true", help="Compatibility flag (accepted, always uses 1.3B path)")
    parser.add_argument("--num_frames", type=int, default=24, help="Compatibility flag for dataset/video frames")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--no_gpu", action="store_true", help="Force CPU mode")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.latent_dir, exist_ok=True)

    model = MoChALoRALightning(
        learning_rate=args.learning_rate,
    )
    model.load_models()

    base_dataset = VideoRefDataset(args.data_path)
    dataset = LatentDataset(base_dataset, args.latent_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # IMPORTANT FIX
    model_device = next(model.dit.parameters()).device
    print("Training device:", model_device)

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
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        precision="32",
        logger=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, dataloader)

    final_dir = os.path.join(args.output_dir, "lora_final")
    os.makedirs(final_dir, exist_ok=True)

    model.dit.save_pretrained(final_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
