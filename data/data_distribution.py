# Single Jupyter cell or standalone Python script to display class distribution as a table and save it as an image
# Function renamed to class_distribution, with automated title formatting
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

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
