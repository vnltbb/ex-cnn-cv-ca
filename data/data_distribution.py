# Single Jupyter cell or standalone Python script to display class distribution as a table and save it as an image
# Function renamed to class_distribution, with automated title formatting
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from IPython.display import display

# Function definition
def class_distribution(train_gen, val_gen, test_gen, config, save_path=None):
    """
    Display and optionally save class distribution table for train, val, test generators.

    config: dict - should contain 'experiment_id'
    save_path: Optional[str] - file path to save the table image
    """
    # Prepare datasets
    datasets = {
        "Train": train_gen,
        "Validation": val_gen,
        "Test": test_gen
    }

    # Map class indices to labels
    class_labels = {v: k for k, v in train_gen.class_indices.items()}

    # Count images per class for each dataset
    counts = {name: Counter(gen.classes) for name, gen in datasets.items()}

    # Build DataFrame
    df = pd.DataFrame(
        {dataset: [counts[dataset].get(i, 0) for i in range(len(class_labels))]
        for dataset in datasets
        },
        index=[class_labels[i] for i in range(len(class_labels))]
    )
    df.index.name = 'Class'

    # Generate title from config
    title = f"{config['experiment_id']}_dataset_distribution"

    # Display DataFrame in Jupyter
    from IPython.display import display
    print(title)
    display(df)

    # Save as image if requested
    if save_path:
        fig, ax = plt.subplots(figsize=(8, 0.6 * len(df) + 1.5))
        ax.axis('off')
        ax.text(0, 1, title, fontsize=14, weight='bold', va='top')
        tbl = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            rowLabels=df.index,
            cellLoc='center',
            loc='center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1, 1.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved class distribution table to '{save_path}'")

def class_distribution_cv(fold_train_dfs, fold_val_dfs, test_df, config, save_path=None):
    """
    교차 검증의 각 폴드별 train/val 및 test 데이터셋의 클래스 분포를 표로 생성하고 저장합니다.
    (행: 데이터셋/폴드, 열: 클래스)

    :param fold_train_dfs: 각 폴드의 훈련 데이터프레임 리스트.
    :param fold_val_dfs: 각 폴드의 검증 데이터프레임 리스트.
    :param test_df: 테스트 데이터프레임.
    :param config: 'experiment_id'를 포함한 설정 딕셔너리.
    :param save_path: 표를 이미지로 저장할 경로.
    """
    distribution_data = {}
    class_names = sorted(test_df['class'].unique())

    # 각 폴드의 train/val 분포 계산
    for i, (train_df, val_df) in enumerate(zip(fold_train_dfs, fold_val_dfs)):
        train_counts = train_df['class'].value_counts().reindex(class_names, fill_value=0)
        val_counts = val_df['class'].value_counts().reindex(class_names, fill_value=0)
        distribution_data[f'Fold {i+1} (Train)'] = train_counts
        distribution_data[f'Fold {i+1} (Val)'] = val_counts

    # Test 데이터 분포 계산
    distribution_data['Test'] = test_df['class'].value_counts().reindex(class_names, fill_value=0)

    # DataFrame 생성
    df = pd.DataFrame(distribution_data).transpose()
    df.index.name = 'Used-Dataset' # 인덱스 이름도 적절하게 변경
    
    title = f"{config['experiment_id']}_CV_Dataset_Distribution"
    
    # 콘솔 및 Jupyter 환경에 표 출력
    print(title)
    display(df)

    # 표를 이미지로 저장
    if save_path:
        fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 1.5), len(df) * 0.4 + 1.5))
        ax.axis('off')
        ax.set_title(title, fontsize=14, weight='bold', pad=20)
        
        tbl = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\n✅ CV 데이터 분포 표를 '{save_path}'에 저장했습니다.")
        plt.show()