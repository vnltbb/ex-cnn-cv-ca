import os
import json

def save_results(model_name, history, cm, class_names, report, result_dir):
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)