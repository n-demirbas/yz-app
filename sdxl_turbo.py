from diffusers import AutoPipelineForText2Image
import torch

print("Model yükleniyor...")
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.bfloat16,
)
pipe.to("cpu")
print("✅ Model hazır!")

prompt = "A cat sitting on a beach, sunset, photorealistic"

print("Görsel üretiliyor...")
image = pipe(
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=0.0
).images[0]

image.save("test_output.png")
print("✅ test_output.png kaydedildi!")