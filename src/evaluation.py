import numpy as np

def print_scores(name, scores):
    print(f"\n=== {name} ===")
    print("Precision:", round(np.mean(scores["test_precision"]), 3))
    print("Recall:", round(np.mean(scores["test_recall"]), 3))
    print("F1:", round(np.mean(scores["test_f1"]), 3))
    print("ROC AUC:", round(np.mean(scores["test_roc_auc"]), 3))
