# utils/preprocessing.py
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance

# === Fungsi ELA ===
def ela_image(image_path, quality=90):
    original_image = Image.open(image_path).convert('RGB')
    resaved_path = 'resaved_temp.jpg'
    original_image.save(resaved_path, 'JPEG', quality=quality)
    resaved_image = Image.open(resaved_path)

    ela_img = ImageChops.difference(original_image, resaved_image)
    extrema = ela_img.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1

    ela_img = ImageEnhance.Brightness(ela_img).enhance(scale)
    ela_img = ela_img.resize((128, 128))
    return np.array(ela_img)

# === SRM Filters ===
def get_srm_filters():
    kernels = []
    for _ in range(30):
        kernel = np.zeros((7, 7), np.float32)
        kernel[3, 3] = -1
        kernel[2:5, 2:5] = 1 / 16
        kernels.append(kernel)
    return kernels

srm_filters = get_srm_filters()

def apply_high_pass_filter(image, kernel):
    return cv2.filter2D(image, -1, kernel)

# === Fungsi SRM ===
def srm_average(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    noise_imgs = [apply_high_pass_filter(img, k) for k in srm_filters]
    avg_noise = np.mean(noise_imgs, axis=0).astype(np.uint8)
    return avg_noise