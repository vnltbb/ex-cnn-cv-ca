from tensorflow.keras.optimizers import Adam, AdamW

def get_optimizer(config, name='adam'):
    
    """ 옵티마이저를 확장하고 싶다면 이런 형식으로
    def get_optimizer(config, name='adam'):
    lr = config.get('learning_rate', 1e-3)  # 기본값 지정
    weight_decay = config.get('weight_decay', 0.0)

    if name.lower() == 'adamw':
        return AdamW(learning_rate=lr, weight_decay=weight_decay)
    return Adam(learning_rate=lr)
"""
    
    if name.lower() == 'adamw':
        return AdamW(learning_rate=config['learning_rate'])
    return Adam(learning_rate=config['learning_rate'])
