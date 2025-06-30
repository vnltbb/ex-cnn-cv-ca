try:
    from tensorflow.keras.optimizers import Adam, AdamW
except ImportError:
    try:
        from keras.optimizers import Adam, AdamW
    except ImportError:
        Adam = None
        AdamW = None

def get_optimizer(optimizer_name='adamw', learning_rate=1e-3, weight_decay=0.0):
    
    if isinstance(weight_decay, str):
        weight_decay = float(weight_decay)
    name = optimizer_name.lower()
    
    if name == 'adamw':
        if AdamW is None:
            print("⚠️ AdamW를 찾을 수 없습니다. 기본 Adam으로 대체합니다.")
            return Adam(learning_rate=learning_rate)
        return AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    return Adam(learning_rate=learning_rate)

