from tensorflow.keras.optimizers import Adam, AdamW

def get_optimizer(optimizer_name='adamw', learning_rate=1e-3, weight_decay=0.0):
    
    if isinstance(weight_decay, str):
        weight_decay = float(weight_decay)
    
    name = optimizer_name.lower()
    
    if name == 'adamw':
            return AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    elif name == 'adam':
            return Adam(learning_rate=learning_rate)


