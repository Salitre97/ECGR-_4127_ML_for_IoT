import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
        return None

def calculate_confusion_matrix(df):
    tp = ((df['true_class'] == 1) & (df['prediction'] == 1)).sum()
    fp = ((df['true_class'] == 0) & (df['prediction'] == 1)).sum()
    tn = ((df['true_class'] == 0) & (df['prediction'] == 0)).sum()
    fn = ((df['true_class'] == 1) & (df['prediction'] == 0)).sum()
    return tp, fp, tn, fn

def calculate_precision_recall(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall

def plot_roc_curve(df, save=False):
    fpr, tpr, thresholds = roc_curve(df['true_class'], df['model_output'])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    if save:
        plt.savefig("roc_curve.png")
    else:
        plt.show()


def find_min_false_positive_rate(df, threshold=0.5):
    fpr, tpr, thresholds = roc_curve(df['true_class'], df['model_output'])
    idx = np.argmax(tpr >= 0.9)  # Index where we achieve at least 90% recall
    min_fpr = fpr[idx]
    return min_fpr

def main():
    file_path = "/home/csalitre/school/ecgr-4127/tinyml/datasets/hw4_data.csv"  # Replace with the path to your CSV file
    df = load_dataset(file_path)
    if df is not None:
        tp, fp, tn, fn = calculate_confusion_matrix(df)
        precision, recall = calculate_precision_recall(tp, fp, fn)
        min_fpr = find_min_false_positive_rate(df)

        with open("results.md", "w") as f:
            f.write("# Results\n")
            f.write("## Confusion Matrix\n")
            f.write("|            | Predicted Positive | Predicted Negative |\n")
            f.write("|------------|-------------------|-------------------|\n")
            f.write(f"| Actual Positive | {tp} | {fn} |\n")
            f.write(f"| Actual Negative | {fp} | {tn} |\n\n")
            
            f.write("## Metrics\n")
            f.write(f"- Precision: {precision}\n")
            f.write(f"- Recall: {recall}\n")
            f.write(f"- Minimum False Positive Rate for at least 90% recall: {min_fpr}\n\n")
            
            f.write("## ROC Curve\n")
            f.write("![ROC Curve](roc_curve.png)\n")

        plot_roc_curve(df, save=True)
        print("Results saved to 'results.md'")


if __name__ == "__main__":
    main()