def train_model(model, train_gen, val_gen, epochs, callbacks, optimizer):
    model.build(input_shape=(None,) + train_gen.image_shape)
    
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=epochs, 
        callbacks=callbacks,
        verbose=1
    )
    return history
