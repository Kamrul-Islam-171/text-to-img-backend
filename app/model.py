import torch
from diffusers import StableDiffusionPipeline

MODEL_ID = "dreamlike-art/dreamlike-photoreal-2.0"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
)

pipe = pipe.to(device)

print("Model loaded successfully!")