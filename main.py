import os
import torch
from fastapi import FastAPI, UploadFile, File, Form
from io import BytesIO
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline
import numpy as np

app = FastAPI()

# Load the inpainting model from Hugging Face
model_name = "mrcuddle/urpm-inpainting"
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Enable memory-efficient attention if xFormers is installed
try:
    import xformers
    pipe.enable_xformers_memory_efficient_attention()
except ImportError:
    pass

@app.post("/inpaint/")
async def inpaint_image(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(None)
):
    try:
        # Load image and mask
        img_data = await image.read()
        mask_data = await mask.read()
        
        img = Image.open(BytesIO(img_data)).convert("RGB")
        mask = Image.open(BytesIO(mask_data)).convert("RGB")
        
        # Inpainting using the model
        generator = torch.manual_seed(42)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img,
            mask_image=mask,
            generator=generator
        ).images[0]

        # Save result to buffer and return as image response
        img_byte_arr = BytesIO()
        result.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        return {"image": img_byte_arr.read()}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
