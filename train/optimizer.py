from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import AdamW

def get_optimizer(optimizer_name='adamw', learning_rate=1e-3, weight_decay=0.0):
    """
    선택된 optimizer 이름에 따라 Adam 또는 AdamW 인스턴스를 반환함.

    Args:
        optimizer_name (str): 'adam' 또는 'adamw'
        learning_rate (float): 학습률
        weight_decay (float): AdamW에서만 사용됨

    Returns:
        tf.keras.optimizers.Optimizer 인스턴스
    """
    name = optimizer_name.lower()
    if name == 'adamw':
        return AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    return Adam(learning_rate=learning_rate)

