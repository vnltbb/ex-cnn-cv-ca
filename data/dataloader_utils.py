import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

preprocess_map = {
    'MobileNetV2': mobilenet_preprocess,
    'EfficientNetB0': efficientnet_preprocess,
    'ResNet50': resnet_preprocess,
    'DenseNet121': densenet_preprocess,
}

def create_dataframe_from_dir(data_dir):
    """
    폴더 구조에서 이미지 경로와 라벨을 담은 데이터프레임 생성
    """
    image_paths = []
    labels = []
    
    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        image_files = glob.glob(os.path.join(cls_path, '*'))
        image_paths.extend(image_files)
        labels.extend([cls] * len(image_files))

    df = pd.DataFrame({'filename': image_paths, 'class': labels})
    return df

def get_generators_from_df(df, model_name, input_shape, batch_size, augmentations, shuffle=True):
    """
    데이터프레임으로부터 ImageDataGenerator 생성
    """
    preprocess_func = preprocess_map[model_name]

    
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )
    
    generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='filename',
        y_col='class',
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=shuffle,
        directory=None
    )
    return generator