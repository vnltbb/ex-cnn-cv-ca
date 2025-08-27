import os, shutil, numpy as np
from sklearn.model_selection import train_test_split

class_image_counts = []

def data_count(original_dataset_dir):
    global class_image_counts
    class_image_counts = {}

    for cls in os.listdir(original_dataset_dir):
        cls_path = os.path.join(original_dataset_dir, cls)
        if os.path.isdir(cls_path):
            images = [img for img in os.listdir(cls_path) if img.endswith('.jpg')]
            class_image_counts[cls] = len(images)
    min_count = min(class_image_counts.values())
    return class_image_counts, images, min_count


def split_dataset_by_class(original_dataset_dir, min_count, images, base_output_dir, train_ratio, val_ratio, test_ratio, seed=42):
    np.random.seed(seed)
    
    for cls in os.listdir(original_dataset_dir):
        cls_path = os.path.join(original_dataset_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        images = [img for img in os.listdir(cls_path) if img.endswith('.jpg')]
        if len(images) == 0:
            print(f"⚠️ 경고: 클래스 '{cls}'에 이미지가 없습니다. 건너뜁니다.")
            continue
        
        np.random.shuffle(images)
        images = images[:min_count]  # 최소 개수로 자르기
        
        # 데이터셋 분할
        train_imgs, temp_imgs = train_test_split(images, train_size=train_ratio, random_state=seed)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_ratio/(val_ratio+test_ratio), random_state=seed)
        
        for category, category_imgs in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            save_path = os.path.join(base_output_dir, category, cls)
            os.makedirs(save_path, exist_ok=True)
            for img in category_imgs:
                shutil.copy(os.path.join(cls_path, img), os.path.join(save_path, img))
                
    print("✅ 클래스별 동일 개수로 데이터셋 분할 완료. 경로:", base_output_dir)


def extract_balanced_dataset(original_dataset_dir, min_count, base_output_dir, seed=42):
    np.random.seed(seed)
    
    for cls in os.listdir(original_dataset_dir):
        cls_path = os.path.join(original_dataset_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        images = [img for img in os.listdir(cls_path) if img.endswith('.jpg')]
        if len(images) == 0:
            print(f"⚠️ 클래스 '{cls}'에 이미지가 없습니다. 건너뜁니다.")
            continue

        # 이미지 랜덤 선정 (min_count개)
        np.random.shuffle(images)
        selected_images = images[:min_count]

        # 저장 경로: dataset/클래스명/
        save_path = os.path.join(base_output_dir, cls)
        os.makedirs(save_path, exist_ok=True)

        for img in selected_images:
            shutil.copy(os.path.join(cls_path, img), os.path.join(save_path, img))
    
    print("✅ 클래스별 동일 개수로 balanced dataset 생성 완료. 경로:", base_output_dir)

def split_dataset_for_cv(original_dataset_dir, min_count, base_output_dir, train_val_ratio, seed=42):
    """
    교차 검증을 위해 전체 데이터셋을 'train_val'과 'test'로 분할합니다.
    'train_val' 디렉토리에는 K-Fold에 사용될 모든 훈련/검증 데이터가 포함됩니다.

    :param original_dataset_dir: 원본 데이터셋 경로
    :param min_count: 클래스별로 사용할 최소 이미지 수 (밸런싱)
    :param base_output_dir: 분할된 데이터가 저장될 경로 (e.g., 'data/split_data/seed42')
    :param train_val_ratio: 전체 데이터 중 'train_val' 세트가 차지할 비율
    :param seed: 재현성을 위한 시드값
    """
    np.random.seed(seed)
    
    # 각 클래스 디렉토리를 순회하며 분할 및 복사 수행
    for cls in os.listdir(original_dataset_dir):
        cls_path = os.path.join(original_dataset_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        # 클래스별 이미지 목록을 가져와서 최소 개수로 맞춤
        images = [img for img in os.listdir(cls_path) if img.endswith('.jpg')]
        if len(images) == 0:
            print(f"⚠️ 경고: 클래스 '{cls}'에 이미지가 없습니다. 건너뜁니다.")
            continue
        
        np.random.shuffle(images)
        images = images[:min_count]
        
        # 'train_val'과 'test' 세트로 분할
        train_val_imgs, test_imgs = train_test_split(images, train_size=train_val_ratio, random_state=seed)
        
        # 해당 디렉토리에 파일 복사
        for category, category_imgs in zip(['train_val', 'test'], [train_val_imgs, test_imgs]):
            save_path = os.path.join(base_output_dir, category, cls)
            os.makedirs(save_path, exist_ok=True)
            for img in category_imgs:
                shutil.copy(os.path.join(cls_path, img), os.path.join(save_path, img))
                
    print(f"✅ 교차 검증용 데이터셋 분할 완료. 경로: {base_output_dir}")