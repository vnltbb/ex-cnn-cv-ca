from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
from tensorflow.keras.callbacks import CSVLogger

def get_callbacks(model_name, save_dir, patience):
    
    early = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f'{save_dir}/{model_name}.h5', save_best_only=True)
    redu = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=patience//2, min_lr=1e-6)
    
    log= CSVLogger(f'{save_dir}/{model_name}.csv', append=True)
    return [checkpoint, early, redu, log]
