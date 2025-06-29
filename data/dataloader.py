from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50, DenseNet121
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from data.color_agu import apply_color_aug
import numpy as np

preprocess_map = {
    'MobileNetV2': mobilenet_preprocess,
    'EfficientNetB0': efficientnet_preprocess,
    'ResNet50': resnet_preprocess,
    'DenseNet121': densenet_preprocess,
}

def undo_preprocessing(image, model_name):
    """모델별 전처리 해제를 위한 함수"""
    if model_name == 'ResNet50':
        # ResNet: [-123.68, -116.779, -103.939] mean subtraction → add mean
        image = image + [123.68, 116.779, 103.939]
    elif model_name == 'EfficientNetB0':
        # EfficientNet: [-1, 1] → 복원
        image = (image + 1) * 127.5
    elif model_name == 'MobileNetV2':
        # MobileNetV2: [-1, 1] → 복원
        image = (image + 1) * 127.5
    elif model_name == 'DenseNet121':
        # DenseNet: mean=[103.94, 116.78, 123.68] (BGR) → OpenCV 기반이므로 복원이 까다롭지만 일단 간단하게
        image = image + [103.94, 116.78, 123.68]
    else:
        # 그 외 모델은 normalize된 [0,1] 범위라고 가정
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def show_samples(gen, model_name='resnet', n_per_class=5):
    """
    클래스별로 n_per_class 장씩 균등하게 이미지를 시각화한다.
    모델 전처리에 따라 복원해서 사람이 보기 좋게 표시한다.
    """
    g_dict = gen.class_indices
    classes = list(g_dict.keys())
    images, labels = next(gen)

    if isinstance(images[0], str):
        print("❌ 이미지가 배열이 아니라 경로(str)입니다.")
        print("예: ", images[0])
        return

    # 클래스별 인덱스 수집
    idx_per_class = {cls: [] for cls in range(len(classes))}
    for idx, label in enumerate(labels):
        class_idx = np.argmax(label)
        if len(idx_per_class[class_idx]) < n_per_class:
            idx_per_class[class_idx].append(idx)
        if all(len(v) == n_per_class for v in idx_per_class.values()):
            break

    plt.figure(figsize=(20, 2.5 * len(classes)))
    plot_idx = 1
    for class_idx, idxs in idx_per_class.items():
        for i in idxs:
            plt.subplot(len(classes), n_per_class, plot_idx)
            image = undo_preprocessing(images[i], model_name)
            plt.imshow(image)
            plt.title(classes[class_idx], color='blue', fontsize=12)
            plt.axis('off')
            plot_idx += 1
    plt.tight_layout()
    plt.show()


def get_generators(model_name, input_shape=(224, 224, 3), batch_size=None, data_dir=None, augmentations=None):
    if data_dir is None:
        raise ValueError("❌ data_dir 값을 지정해야 합니다.")
    
    if augmentations is None:
        augmentations = []

    preprocess_func = preprocess_map[model_name]

    def custom_preprocessing(img):
        img = preprocess_func(img)
        img = apply_color_aug(img, augmentations)
        return img
    
    # folder 
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # train generator with augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )
    
    train_gen = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
    )

    # validation generator
    val_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing,
        validation_split=0.2,
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
    )

    # test generator (no augmentation)
    test_gen = ImageDataGenerator(preprocessing_function=custom_preprocessing).flow_from_directory(
        test_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    )

    return train_gen, val_gen, test_gen
