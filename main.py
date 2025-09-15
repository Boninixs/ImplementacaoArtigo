from src.preprocess import load_data, preprocess_data
from src.models import train_models
from src.evaluation import print_scores

# 1. carregar dataset
X, y = load_data("data/synthetic_metrics.csv")

# 2. pré-processar
X_pre, y_pre, scaler = preprocess_data(X, y)

# 3. treinar modelos
scores_svm, scores_lr = train_models(X_pre, y_pre)

# 4. mostrar resultados
print_scores("SVM", scores_svm)
print_scores("Logistic Regression", scores_lr)

