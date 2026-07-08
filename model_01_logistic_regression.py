import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from models.data_loader import load_data, flatten_normalise, save_result, print_results

X_train, y_train, X_test, y_test, class_names = load_data()
X_train_f = flatten_normalise(X_train)
X_test_f  = flatten_normalise(X_test)

X_sub, _, y_sub, _ = train_test_split(
    X_train_f, y_train, train_size=30000, stratify=y_train, random_state=42)

print("Training Logistic Regression on 30k stratified samples...")
t0 = time.time()
model = LogisticRegression(max_iter=500, C=0.5, solver="lbfgs", n_jobs=-1)
model.fit(X_sub, y_sub)
t_train = time.time() - t0

t0 = time.time()
preds = model.predict(X_test_f)
t_infer = time.time() - t0

acc = accuracy_score(y_test, preds)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted")
cm = confusion_matrix(y_test, preds)

path = save_result("1. Logistic Regression", acc, prec, rec, f1, t_train, t_infer, cm)
print_results("1. Logistic Regression", acc, f1, t_train, t_infer)
print(f"Result saved → {path}")
