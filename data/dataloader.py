import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50, DenseNet121
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from sklearn.model_selection import train_test_split
from data.color_agu import apply_color_aug


preprocess_map = {
    'MobileNetV2': mobilenet_preprocess,
    'EfficientNetB0': efficientnet_preprocess,
    'ResNet50': resnet_preprocess,
    'DenseNet121': densenet_preprocess,
}


def get_generators(model_name, input_shape=(224, 224, 3), batch_size=None, data_dir=None,train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, augmentations=None):
    if data_dir is None:
        raise ValueError("❌ data_dir 값을 지정해야 합니다.")
    if augmentations is None:
        augmentations = []

    preprocess_func = preprocess_map[model_name]

    def custom_preprocessing(img):
        img = preprocess_func(img)
        img = apply_color_aug(img, augmentations)
        return img
    
    # 이미지 경로 및 라벨 수집
    image_paths = []
    labels = []
    
    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        imgs = [f for f in os.listdir(cls_path) if f.endswith(".jpg")]
        for img in imgs:
            image_paths.append(os.path.join(cls_path, img))
            labels.append(cls)

    df = pd.DataFrame({'filename': image_paths, 'class': labels})
    
    # 클래스별 stratified split
    train_df, temp_df = train_test_split(df, stratify=df['class'], train_size=train_ratio, random_state=seed)
    val_df, test_df = train_test_split(
        temp_df,
        stratify=temp_df['class'],
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed
    )
    
    # generator setup
    train_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )
    
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
    
    # generator 생성
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filename',
        y_col='class',
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filename',
        y_col='class',
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    )

    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filename',
        y_col='class',
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    )

    return train_gen, val_gen, test_gen
