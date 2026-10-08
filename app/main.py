from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .model import pipe, device
from .schemas import GenerateRequest

import os
import uuid


app = FastAPI(
    title="Text to Image API",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


GENERATED_DIR = "generated_images"

os.makedirs(GENERATED_DIR, exist_ok=True)


app.mount(
    "/generated",
    StaticFiles(directory=GENERATED_DIR),
    name="generated"
)


@app.get("/")
def root():
    return {
        "message": "Text-to-Image API is running",
        "device": device
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device
    }


@app.post("/generate")
def generate_image(request: GenerateRequest):

    try:

        result = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            num_images_per_prompt=1
        )

        image = result.images[0]

        filename = f"{uuid.uuid4()}.png"

        filepath = os.path.join(
            GENERATED_DIR,
            filename
        )

        image.save(filepath)

        return {
            "success": True,
            "filename": filename,
            "image_url": f"/generated/{filename}"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )