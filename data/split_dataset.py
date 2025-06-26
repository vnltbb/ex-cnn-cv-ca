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
