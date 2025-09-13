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

def plot_metrics_text(metrics_dict, experiment_id=None, save_path=None, cv_info=None):
    """
    experiment_id 및 주요 성능 지표를 텍스트로 시각화
    metrics_dict: dict - accuracy, precision, recall, f1_score 포함
    cv_info: dict - {'use_cv': bool, 'cv_type': str} 교차 검증 정보
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis('off')

    text_lines = []
    if experiment_id:
        text_lines.append(f"Experiment ID:\n{experiment_id}\n")
    
        if cv_info and cv_info.get('use_cv'):
            cv_type = cv_info.get('cv_type', 'N/A').replace('_', ' ').title()
            text_lines.append(f"Cross-Validation:\n{cv_type}\n")
    
    for key in ["accuracy", "precision", "recall", "f1_score"]:
        val = metrics_dict.get(key, None)
        if val is not None:
            text_lines.append(f"{key.capitalize()}: {val:.4f}")

    full_text = "\n".join(text_lines)
    ax.text(0.05, 0.95, full_text, fontsize=12, verticalalignment='top')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_cross_val_metrics(results_df, experiment_id, save_path=None):
    """
    교차 검증 결과를 박스 플롯으로 시각화하여 폴드별 성능 분포를 보여줍니다.

    :param results_df: 폴드별 결과가 담긴 pandas DataFrame.
    :param experiment_id: 그래프 제목에 사용될 실험 ID.
    :param save_path: 그래프를 이미지 파일로 저장할 경로.
    """
    # 시각화할 지표만 선택 (fold 번호 제외)
    metrics_to_plot = results_df.drop(columns=['fold'])
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 박스 플롯 생성
    sns.boxplot(data=metrics_to_plot, ax=ax, palette='viridis')
    
    # 평균값을 점으로 표시
    means = metrics_to_plot.mean()
    for i, metric in enumerate(metrics_to_plot.columns):
        ax.plot(i, means[i], 'rD', markersize=8, label='Mean' if i == 0 else "")

    ax.set_title(f'Cross-Validation Metrics Distribution\n({experiment_id})', fontsize=16)
    ax.set_ylabel('Score', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"✅ 교차 검증 지표 그래프를 '{save_path}'에 저장했습니다.")
        
    plt.show()


def plot_combined_history(histories, experiment_id, save_path=None):
    """
    여러 폴드의 Keras History 객체를 하나로 합쳐 학습 그래프를 그립니다.
    각 폴드의 경계는 점선으로 표시합니다.

    :param histories: Keras History 객체들의 리스트.
    :param experiment_id: 그래프 제목에 사용될 실험 ID.
    :param save_path: 그래프를 이미지 파일로 저장할 경로.
    """
    # 모든 history에서 accuracy, val_accuracy, loss, val_loss를 추출하여 하나로 합침
    acc = [item for sublist in [h.history['accuracy'] for h in histories] for item in sublist]
    val_acc = [item for sublist in [h.history['val_accuracy'] for h in histories] for item in sublist]
    loss = [item for sublist in [h.history['loss'] for h in histories] for item in sublist]
    val_loss = [item for sublist in [h.history['val_loss'] for h in histories] for item in sublist]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'Combined Training History\n({experiment_id})', fontsize=16)

    # Accuracy 그래프
    ax1.plot(acc, label='Train Accuracy', color='royalblue')
    ax1.plot(val_acc, label='Validation Accuracy', color='darkorange')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.set_title('Accuracy over Folds')

    # Loss 그래프
    ax2.plot(loss, label='Train Loss', color='royalblue')
    ax2.plot(val_loss, label='Validation Loss', color='darkorange')
    ax2.set_xlabel('Total Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.set_title('Loss over Folds')

    # 각 폴드 경계에 점선과 텍스트 추가
    epoch_cumsum = 0
    for i, h in enumerate(histories):
        num_epochs = len(h.history['accuracy'])
        epoch_cumsum += num_epochs
        if i < len(histories) - 1: # 마지막 폴드 제외
            for ax in [ax1, ax2]:
                ax.axvline(x=epoch_cumsum - 0.5, color='grey', linestyle='--', linewidth=1)
                ax.text(epoch_cumsum - num_epochs / 2, 0.5, f'Fold {i+1}', transform=ax.get_xaxis_transform(), 
                        ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"✅ 통합 History 그래프를 '{save_path}'에 저장했습니다.")
    
    plt.show()