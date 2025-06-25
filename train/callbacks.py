from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def get_callbacks(model_name, save_dir, patience):
    early = EarlyStopping(patience=patience, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f'{save_dir}/{model_name}.h5', save_best_only=True)
    redu = ReduceLROnPlateau(patience=patience//2)
    return [checkpoint, early, redu]
