from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0, DenseNet121
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

def get_model(backbone_name, input_shape):
    if backbone_name == 'ResNet50':
        return ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    elif backbone_name == 'MobileNetV2':
        return MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    elif backbone_name == 'EfficientNetB0':
        return EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    elif backbone_name == 'DenseNet121':
        return DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape)
    else:
        raise ValueError(f"❌ 지원되지 않는 백본입니다: {backbone_name}")

def build_model(backbone_name, input_shape, num_classes, dropout_rate=0.5):
    input_tensor = Input(shape=input_shape)
    
    backbone = get_model(backbone_name, input_shape)
    x = base_model(input_tensor, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=backbone.input, outputs=outputs, name=f"{backbone_name}_custom")
    return model
