from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from PIL import Image
import torch
import numpy as np
import io

app = FastAPI()

# BiRefNet
print("BiRefNet yükleniyor...")
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet",
    trust_remote_code=True,
    torch_dtype=torch.float32
)
birefnet.eval()
print("✅ BiRefNet hazır!")

birefnet_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Real-ESRGAN
print("Real-ESRGAN yükleniyor...")
esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                       num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    model=esrgan_model,
    tile=128,
    tile_pad=10,
    pre_pad=0,
    half=False
)
print("✅ Real-ESRGAN hazır!")

@app.get("/")
def root():
    return {
        "status": "API çalışıyor 🚀",
        "endpoints": ["/remove-bg", "/upscale"]
    }


@app.post("/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    original_size = image.size

    input_tensor = birefnet_transform(image).unsqueeze(0).float()

    with torch.no_grad():
        preds = birefnet(input_tensor)[-1].sigmoid().cpu()

    mask = transforms.functional.resize(preds[0], [original_size[1], original_size[0]])
    mask = mask.squeeze().numpy()

    image_np = np.array(image)
    alpha = (mask * 255).astype(np.uint8)
    rgba = np.dstack((image_np, alpha))
    result = Image.fromarray(rgba, "RGBA")

    output = io.BytesIO()
    result.save(output, format="PNG")
    output.seek(0)
    return StreamingResponse(output, media_type="image/png")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)