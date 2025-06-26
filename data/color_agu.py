import cv2
import numpy as np
import random

def adjust_brightness(img, delta=0.2):
    """이미지의 밝기를 랜덤하게 조절"""
    factor = 1.0 + random.uniform(-delta, delta)
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def adjust_contrast(img, delta=0.2):
    """이미지의 대비를 랜덤하게 조절"""
    factor = 1.0 + random.uniform(-delta, delta)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

def adjust_saturation(img, factor_range=(0.8, 1.2)):
    """채도를 주어진 범위에서 랜덤하게 조절"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    factor = random.uniform(*factor_range)
    hsv[..., 1] *= factor
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def color_jitter(img, brightness=0.2, contrast=0.2, saturation_range=(0.8, 1.2)):
    """밝기, 대비, 채도 랜덤 변화"""
    img = adjust_brightness(img, brightness)
    img = adjust_contrast(img, contrast)
    img = adjust_saturation(img, saturation_range)
    return img

def apply_colormap(img, cmap=cv2.COLORMAP_JET):
    """흑백 변환 후 컬러맵 적용"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    colored = cv2.applyColorMap(gray, cmap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

def apply_color_aug(img, aug_list):
    """선택된 aug 기법들을 순차적으로 적용"""
    if 'color_jitter' in aug_list:
        img = color_jitter(img)
    if 'colormap' in aug_list:
        img = apply_colormap(img)
    else:
        img = img
    
    return img
