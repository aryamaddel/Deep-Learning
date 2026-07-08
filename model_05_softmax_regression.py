import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from models.data_loader import load_data, flatten_normalise, save_result, print_results

device = torch.device("cuda" if torch.cuda.is_available() else "mps"
                      if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

X_train, y_train, X_test, y_test, class_names = load_data()
X_train_f = torch.tensor(flatten_normalise(X_train))
X_test_f  = torch.tensor(flatten_normalise(X_test))
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test, dtype=torch.long)

batch_size = 256
train_loader = DataLoader(TensorDataset(X_train_f, y_train_t), batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_f, y_test_t), batch_size=batch_size)

class SoftmaxRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
    def forward(self, x):
        return self.fc(x)

model = SoftmaxRegression().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Training Softmax Regression (full 60k, 10 epochs)...")
t0 = time.time()
for epoch in range(10):
    model.train()
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
t_train = time.time() - t0

model.eval()
all_preds = []
t0 = time.time()
with torch.no_grad():
    for Xb, _ in test_loader:
        preds = model(Xb.to(device)).argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
t_infer = time.time() - t0
preds = np.concatenate(all_preds)

acc = accuracy_score(y_test, preds)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted")
cm = confusion_matrix(y_test, preds)

path = save_result("5. Softmax Regression", acc, prec, rec, f1, t_train, t_infer, cm)
print_results("5. Softmax Regression", acc, f1, t_train, t_infer)
print(f"Result saved → {path}")
