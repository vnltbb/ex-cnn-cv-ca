def save_results(model_name, history, cm, class_names, report):
    result_dir = f"result/{HYPERPARAMS['experiment_id']}"
    os.makedirs(result_dir, exist_ok=True)

    plot_confusion_matrix(cm, class_names, title=f'{model_name} Confusion Matrix',
                          save_path=os.path.join(result_dir, "confusion_matrix.png"))
    plot_train_history(history, title_prefix=model_name,
                       save_path=os.path.join(result_dir, "accuracy_loss.png"))
    with open(os.path.join(result_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)