import os
import json

def save_results(model_name, history, cm, class_names, report, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)