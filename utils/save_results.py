import os
import json

def save_results(model_name, history, cm, class_names, report, save_path):
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)