from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold

def train_models(X, y):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# n_splits= 3, usa o 3 pq o código o csv é pequeno, alterar pra 10 depois. 3 = 2 para treinamento + 1 para teste
    svm = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)
    lr = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)

    scores_svm = cross_validate(svm, X, y, cv=cv, scoring=["precision", "recall", "f1", "roc_auc"])
    scores_lr = cross_validate(lr, X, y, cv=cv, scoring=["precision", "recall", "f1", "roc_auc"])

    return scores_svm, scores_lr


#Precision: "Das minhas previsões positivas, quantas estão corretas?"
#Recall: "De todos os casos positivos reais, quantos eu capturei?"
#F1: Balanceamento entre Precision e Recall
#ROC AUC: "Qual a capacidade geral de discriminar entre classes?"