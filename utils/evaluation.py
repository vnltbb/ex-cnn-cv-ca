import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import cv2

def evaluate_model(model, test_gen):
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes 
    
    return y_true, y_pred, y_prob


def plot_confusion_matrix(cm, class_names, title, save_path=None):
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_train_history(history, save_path=None):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def show_top_misclassified(y_true, y_pred, probs, class_names, generator, model_name, save_dir, top_n=3):
    error_indices = np.where(y_true != y_pred)[0]
    confidence_errors = probs[error_indices, y_pred[error_indices]]
    sorted_idx = error_indices[np.argsort(confidence_errors)[-top_n:][::-1]]

    os.makedirs(save_dir, exist_ok=True)

    for i, idx in enumerate(sorted_idx):
        img_path = generator.filepaths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(
            f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}\nConf: {confidence_errors[i]:.2f}"
        )
        plt.axis('off')

        save_path = os.path.join(save_dir, f"misclassified_{i+1}.png")
        plt.savefig(save_path)
        plt.show()
        