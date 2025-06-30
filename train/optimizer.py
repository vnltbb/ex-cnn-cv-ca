from tensorflow.keras.optimizers import Adam, AdamW

def get_optimizer(optimizer_name='adamw', learning_rate=1e-3, weight_decay=0.0):
    
    if isinstance(weight_decay, str):
        weight_decay = float(weight_decay)
    
    name = optimizer_name.lower()
    
    if name == 'adamw':
        if AdamW is not None:
            return AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        elif Adam is not None:
            print("⚠️ AdamW를 찾을 수 없어 Adam으로 대체합니다.")
            return Adam(learning_rate=learning_rate)
        else:
            raise ImportError("❌ AdamW도 Adam도 import되지 않았습니다. optimizer 설정 확인 필요.")

    elif name == 'adam':
        if Adam is not None:
            return Adam(learning_rate=learning_rate)
        else:
            raise ImportError("❌ Adam이 import되지 않았습니다. optimizer 설정 확인 필요.")

    else:
        raise ValueError(f"❌ 지원하지 않는 optimizer입니다: {optimizer_name}")
    return Adam(learning_rate=learning_rate)

