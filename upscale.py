from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import numpy as np
from PIL import Image

print("Model yükleniyor...")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4)

upsampler = RealESRGANer(
    scale=4,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    model=model,
    tile=128,
    tile_pad=10,
    pre_pad=0,
    half=False
)
print("✅ Model hazır!")

# Test görseli oluştur
test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

print("Upscale yapılıyor...")
output, _ = upsampler.enhance(test_img, outscale=4)
result = Image.fromarray(output)
result.save("upscale_test.png")
print(f"✅ Tamamlandı! Çıktı boyutu: {result.size}")