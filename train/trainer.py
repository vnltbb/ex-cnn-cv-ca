def train_model(model, train_gen, val_gen, epochs, callbacks, optimizer):
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    return history
