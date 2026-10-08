from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=500
    )

    negative_prompt: str | None = None

    num_inference_steps: int = Field(
        default=20,
        ge=1,
        le=50
    )

    guidance_scale: float = Field(
        default=7.5,
        ge=1,
        le=20
    )