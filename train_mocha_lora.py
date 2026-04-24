import os
import torch
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
import argparse

from diffsynth import ModelManager, WanVideoMoChaPipeline
from peft import LoraConfig, inject_adapter_in_model


# =========================
# CPU OPTIMIZATION
# =========================
NUM_THREADS = 12
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(6)
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
torch.backends.mkldnn.enabled = True


# =========================
# LATENT DATASET WRAPPER
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
# MODEL
# =========================
class LightningModelForMoChALoRA(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, lora_rank=8, lora_alpha=16):
        super().__init__()

        self.learning_rate = learning_rate
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        self.pipe = None

    def load_models(self):
        device = torch.device("cpu")

        print(f"Loading models on CPU ({NUM_THREADS} threads)...")

        model_manager = ModelManager(
            torch_dtype=torch.float32,
            device=device
        )

        model_manager.load_models([
            "./models/wan2.1_1.3b/diffusion_pytorch_model.safetensors",
            "./models/wan2.1_1.3b/models_t5_umt5-xxl-enc-bf16.pth",
            "./models/wan2.1_1.3b/Wan2.1_VAE.pth",
        ])

        self.pipe = WanVideoMoChaPipeline.from_model_manager(
            model_manager,
            device=device,
        )

        self.pipe.requires_grad_(False)

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=["q", "k", "v", "o"],
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )

        print("Injecting LoRA...")
        self.pipe.dit = inject_adapter_in_model(lora_config, self.pipe.dit)

        self.pipe.train()

    def training_step(self, batch, batch_idx):
        device = self.device

        # -------------------------
        # LATENT CACHE LOGIC
        # -------------------------
        if batch["needs_cache"][0]:
            video = batch["video"].to(device)

            print(f"[Batch {batch_idx}] Caching latent...")

            with torch.no_grad():
                latents = self.pipe.vae.encode(video, device=device)

            torch.save(latents.cpu(), batch["latent_path"][0])
        else:
            latents = torch.load(batch["latent_path"][0]).to(device)

        # -------------------------
        # DIFFUSION
        # -------------------------
        noise = torch.randn_like(latents)

        t_id = torch.randint(
            0,
            self.pipe.scheduler.num_train_timesteps,
            (1,),
            device=device
        )
        timestep = self.pipe.scheduler.timesteps[t_id]

        noisy_latents = self.pipe.scheduler.add_noise(
            latents,
            noise,
            timestep
        )

        # -------------------------
        # TEXT
        # -------------------------
        with torch.no_grad():
            prompt_emb = self.pipe.encode_prompt("")

        # -------------------------
        # DIT (MAIN COST)
        # -------------------------
        noise_pred = self.pipe.dit(
            noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_emb["encoder_hidden_states"],
            encoder_attention_mask=prompt_emb["encoder_attention_mask"],
        )

        loss = torch.nn.functional.mse_loss(
            noise_pred.float(),
            noise.float()
        )

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

        print(f"Trainable LoRA params: {sum(p.numel() for p in trainable_params):,}")

        return torch.optim.AdamW(trainable_params, lr=self.learning_rate)


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./data/train_data.csv")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--latent_dir", type=str, default="./latents")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=10000)

    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)

    args = parser.parse_args()

    # -------------------------
    # MODEL
    # -------------------------
    model = LightningModelForMoChALoRA(
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    model.load_models()

    # -------------------------
    # DATASET
    # -------------------------
    from inference_mocha import VideoRefDataset

    base_dataset = VideoRefDataset(args.data_path, args)
    dataset = LatentDataset(base_dataset, args.latent_dir)

    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    # -------------------------
    # TRAINER
    # -------------------------
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        max_steps=args.max_steps,
        accelerator="cpu",
        devices=1,
        precision=32,
        log_every_n_steps=1,
    )

    trainer.fit(model, dataloader)

    # -------------------------
    # SAVE LORA
    # -------------------------
    final_dir = os.path.join(args.output_dir, "lora_final")
    os.makedirs(final_dir, exist_ok=True)

    model.pipe.dit.save_pretrained(final_dir)

    torch.save(
        {
            name: p.cpu()
            for name, p in model.pipe.dit.named_parameters()
            if "lora" in name
        },
        os.path.join(args.output_dir, "lora_final.ckpt"),
    )


if __name__ == "__main__":
    main()