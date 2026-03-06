from transformers import AutoModelForImageSegmentation
import torch

print("Model indiriliyor...")
model = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet",
    trust_remote_code=True
)
model.eval()
print("✅ Model başarıyla yüklendi!")