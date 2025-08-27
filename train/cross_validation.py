import os
import time
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

from models.build_model import build_model
from train.optimizer import get_optimizer
from train.callbacks import get_callbacks
from train.trainer import train_model
from utils.evaluation import (
    evaluate_model, 
    plot_confusion_matrix, 
    show_top_misclassified,
    plot_combined_history 
)
from utils.save_results import save_results
from data.data_distribution import class_distribution_cv
from data.dataloader_utils import create_dataframe_from_dir, get_generators_from_df
from data.split_dataset import data_count, split_dataset_for_cv


def run_cross_validation(config):
    """
    먼저 데이터를 train_val / test 세트로 분리한 후,
    train_val 세트에 대해서만 교차 검증을 실행합니다.
    """
    print("✅ 교차 검증을 시작합니다.")

    # --- 1. 설정 값 로드 ---
    seed = config['seed']
    original_dataset_dir = config['original_dataset_dir']
    base_output_dir = config['base_output_dir']
    
    # config의 train_ratio와 val_ratio를 합쳐 CV에 사용할 전체 학습 데이터 비율을 계산
    train_val_ratio = config['train_ratio'] + config['val_ratio']

    # 시드값에 따라 분할된 데이터가 저장될 경로 설정
    split_data_dir = os.path.join(base_output_dir, f"cv_seed{seed}")
    train_val_dir = os.path.join(split_data_dir, 'train_val') # CV에 사용할 데이터 경로
    test_dir = os.path.join(split_data_dir, 'test')

    # --- 2. Train_val / Test 데이터셋 분리 ---
    # train_val 폴더가 없다면 데이터 분할을 새로 수행
    if not os.path.exists(train_val_dir):
        print(f"ℹ️ '{split_data_dir}'에 분할된 데이터가 없어 새로 생성합니다.")
        
        class_image_counts, _, min_count = data_count(original_dataset_dir)
        
        # 새로 추가한 CV용 분할 함수 호출
        split_dataset_for_cv(
            original_dataset_dir=original_dataset_dir,
            min_count=min_count,
            base_output_dir=split_data_dir,
            train_val_ratio=train_val_ratio,
            seed=seed
        )
    else:
        print(f"ℹ️ '{split_data_dir}'에 이미 분할된 데이터가 존재하여 분할을 건너뜁니다.")

    # --- 3. 교차 검증을 위한 데이터프레임 생성 (train_val) ---
    print("교차 검증을 위해 Train_val 데이터를 로드합니다.")
    df = create_dataframe_from_dir(train_val_dir)
    df_test = create_dataframe_from_dir(test_dir)
    
    X = df['filename']
    y = df['class']

    # --- 4. 교차 검증 설정 및 실행 ---
    cv_type = config.get('cross_validation_type', 'stratified_kfold')
    n_splits = config.get('n_splits', 5)
    n_repeats = config.get('n_repeats', 1)
    
    batch_size = config['batch_size']
    input_shape = tuple(config['input_shape'])

    if cv_type == 'stratified_kfold':
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    elif cv_type == 'repeated_stratified_kfold':
        kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    else:
        raise ValueError("❌ 지원되지 않는 교차 검증 유형입니다.")
    
    # 변수 초기화
    all_results = []
    fold_histories = [] 
    total_learning_time = 0.0
    fold_train_dfs, fold_val_dfs = [], []
    
    final_results_dir = f"results/{config['experiment_id']}"
    
    # 교차 검증 루프
    for i, (train_index, val_index) in enumerate(kf.split(X, y)):
        print(f"\n--- Fold {i+1}/{n_splits} 시작 ---")

        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]
        fold_train_dfs.append(train_df)
        fold_val_dfs.append(val_df)

        # 데이터 제너레이터 생성
        train_gen = get_generators_from_df(train_df, config['backbone_name'], input_shape, batch_size, config.get('augmentations'), shuffle=True)
        val_gen = get_generators_from_df(val_df, config['backbone_name'], input_shape, batch_size, config.get('augmentations'), shuffle=False)

        # 모델 빌드 및 컴파일
        model = build_model(
            backbone_name=config['backbone_name'],
            input_shape=input_shape,
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate']
        )
        optimizer = get_optimizer(
            optimizer_name=config['optimizer'],
            learning_rate=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.0)
        )
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # 콜백 설정
        fold_name = f"fold_{i+1}"
        save_dir = os.path.join(final_results_dir, "folds", fold_name) # 하위 폴더 경로 생성
        os.makedirs(save_dir, exist_ok=True)
        
        callbacks = get_callbacks(model_name=fold_name, save_dir=save_dir, patience=config['patience'])

        # 모델 훈련
        start_time = time.time()
        history = train_model(model, train_gen, val_gen, config['epochs'], callbacks, optimizer)
        end_time = time.time()
        total_learning_time += (end_time - start_time)
        
        fold_histories.append(history)

        # 모델 평가
        y_true, y_pred, y_prob = evaluate_model(model, val_gen)
        class_names = list(val_gen.class_indices.keys())
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

        # 결과 저장
        print(f"--- Fold {i+1} 결과 시각화 및 저장 ---")
        # 1. Confusion Matrix 저장
        plot_confusion_matrix(cm, class_names, title=f"Fold {i+1} Confusion Matrix", save_path=os.path.join(save_dir, "confusion_matrix.png"))
        
        # 2. Top 3 오분류 이미지 저장
        show_top_misclassified(y_true, y_pred, y_prob, class_names, generator=val_gen,model_name=fold_name, save_dir=save_dir, top_n=3)

        # 3. classification_report.json 저장
        save_results(fold_name, history, cm, class_names, report, save_dir)
        
        # 폴드별 지표 저장
        all_results.append({
            'fold': i + 1,
            'accuracy': report['accuracy'],
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'macro_f1_score': report['macro avg']['f1-score']
        })

    # --- 5. 최종 결과 요약 및 저장 ---
    results_df = pd.DataFrame(all_results)
    avg_metrics = results_df.mean().to_dict()
    std_metrics = results_df.std().to_dict()
    
    print("\n--- 교차 검증 최종 결과 ---")
    print(f"평균 정확도: {avg_metrics.get('accuracy', 0):.4f} ± {std_metrics.get('accuracy', 0):.4f}")
    print(f"평균 Macro F1-score: {avg_metrics.get('macro_f1_score', 0):.4f} ± {std_metrics.get('macro_f1_score', 0):.4f}")
    print(f"총 학습 시간: {total_learning_time / 60:.2f} 분 ({total_learning_time:.2f} 초)")
    
    
    os.makedirs(final_results_dir, exist_ok=True)
    # 데이터 분포 표 생성 및 저장
    class_distribution_cv(
        fold_train_dfs,
        fold_val_dfs,
        df_test,
        config,
        save_path=os.path.join(final_results_dir, "cv_data_distribution.png")
    )
    
    # 통합 History 그래프 저장
    plot_combined_history(fold_histories, config['experiment_id'], save_path=os.path.join(final_results_dir, "combined_history_graph.png"))
    final_results = {
        "individual_folds": all_results,
        "mean_metrics": avg_metrics,
        "std_metrics": std_metrics,
        "config": config
    }
    with open(os.path.join(final_results_dir, "cross_val_results.json"), "w") as f:
        json.dump(final_results, f, indent=4)

    # 최종 결과 이미지로 저장
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    summary_text = (
        f"Cross-Validation Final Results\n"
        f"--------------------------------------\n"
        f"Experiment ID: {config['experiment_id']}\n\n"
        f"Avg Accuracy    : {avg_metrics.get('accuracy', 0):.4f} ± {std_metrics.get('accuracy', 0):.4f}\n"
        f"Avg Macro F1    : {avg_metrics.get('macro_f1_score', 0):.4f} ± {std_metrics.get('macro_f1_score', 0):.4f}\n"
        f"Avg Macro Precision: {avg_metrics.get('macro_precision', 0):.4f} ± {std_metrics.get('macro_precision', 0):.4f}\n"
        f"Avg Macro Recall : {avg_metrics.get('macro_recall', 0):.4f} ± {std_metrics.get('macro_recall', 0):.4f}\n\n"
        f"Total Learning Time: {total_learning_time / 60:.2f} minutes"
    )
    ax.text(0.01, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace', linespacing=1.5)
    plt.tight_layout()
    summary_img_path = os.path.join(final_results_dir, "cross_val_summary.png")
    plt.savefig(summary_img_path, dpi=200, bbox_inches='tight')

    print(f"✅ 교차 검증 완료. 최종 결과는 '{final_results_dir}'에 저장되었습니다.")
    print(f"✅ 최종 결과 요약 이미지를 '{summary_img_path}'에 저장했습니다.")
    return final_results

