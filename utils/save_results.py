import os
import json
import pandas as pd
from typing import Optional

def save_results(model_name, history, cm, class_names, report, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # 히스토리 데이터프레임으로 변환 후 CSV 저장
    df_history = pd.DataFrame(history.history)
    df_history.to_csv(os.path.join(save_dir, "history.csv"), index=False)
    
    # 혼동 행렬 저장
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(save_dir, "confusion_matrix.csv"))

    # 분류 리포트 저장
    with open(os.path.join(save_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"✅ 결과 저장 완료: {save_dir}")