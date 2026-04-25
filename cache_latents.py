import os
import torch
import imageio
import pandas as pd

from PIL import Image
from torchvision.transforms import v2
from einops import rearrange

from diffsynth.models import ModelManager
from diffsynth.pipelines import WanVideoMoChaPipeline
from huggingface_hub import snapshot_download


DATA_PATH = "./data/train_data.csv"
LATENT_DIR = "./latents"

NUM_FRAMES = 24
HEIGHT = 480
WIDTH = 832

os.makedirs(LATENT_DIR, exist_ok=True)


def is_image(path):
    return path.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))


transform = v2.Compose([
    v2.CenterCrop((HEIGHT, WIDTH)),
    v2.Resize((HEIGHT, WIDTH)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5]*3, std=[0.5]*3),
])


def load_image(path):
    img = Image.open(path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0).unsqueeze(2)
    return img


def load_video(path):
    if is_image(path):
        return load_image(path)

    reader = imageio.get_reader(path)
    frames = []

    max_frames = min(NUM_FRAMES, reader.count_frames())

    for i in range(max_frames):
        frame = Image.fromarray(reader.get_data(i))
        frame = transform(frame)
        frames.append(frame)

    reader.close()

    frames = torch.stack(frames, dim=0)
    frames = rearrange(frames, "t c h w -> c t h w")

    frames = frames.unsqueeze(0)
    return frames


print("Loading VAE only...")

snapshot_download(
    repo_id="Wan-AI/Wan2.1-T2V-1.3B",
    local_dir="./models/wan2.1_1.3b"
)

manager = ModelManager(
    torch_dtype=torch.float16,
    device="cpu"
)

manager.load_model(
    "./models/wan2.1_1.3b/Wan2.1_VAE.pth",
    device="cpu",
    torch_dtype=torch.float16
)

pipe = WanVideoMoChaPipeline.from_model_manager(
    manager,
    device="cpu"
)

vae = pipe.vae
vae.to("cpu")
vae.eval()

metadata = pd.read_csv(DATA_PATH)

for idx, row in metadata.iterrows():
    save_path = os.path.join(LATENT_DIR, f"{idx}.pt")

    if os.path.exists(save_path):
        print(f"Skip {idx}")
        continue

    print(f"Encoding {idx}")

    video_path = row["source_video"]   # <-- FIX HERE
    video = load_video(video_path)

    # dtype match fix
    vae_dtype = next(vae.parameters()).dtype
    video = video.to(dtype=vae_dtype)

    with torch.no_grad():
        latent = vae.encode(video, device="cpu")

    torch.save(latent.cpu(), save_path)

print("Latent caching complete.")