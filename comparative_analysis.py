import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

results = []
for fname in sorted(os.listdir(RESULTS_DIR)):
    if fname.endswith(".json"):
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            results.append(json.load(f))

results.sort(key=lambda r: -r["acc"])

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

print("=" * 75)
print("COMPARATIVE ANALYSIS — 8 Models on Fashion MNIST")
print("=" * 75)
print(f"{'Rank':<5s} {'Model':<28s} {'Acc':>7s} {'F1':>7s} {'Train(s)':>9s} {'Infer(s)':>9s}")
print("-" * 65)
for i, r in enumerate(results, 1):
    print(f"{i:<5d} {r['name']:<28s} {r['acc']:.4f}  {r['f1']:.4f}  "
          f"{r['train_time']:>8.1f}  {r['infer_time']:>8.1f}")

# ── Accuracy bar chart ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
names = [r["name"] for r in results]
accs  = [r["acc"] for r in results]
is_nn = ["CNN" in n or "MLP" in n or "Softmax" in n for n in names]
colors = ["#2ecc71" if x else "#3498db" for x in is_nn]
bars = ax.barh(range(len(names)), accs, color=colors, edgecolor="white", height=0.6)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("Test Accuracy")
ax.set_title("Fashion MNIST — Model Comparison", fontsize=14)
ax.set_xlim(0, 1.0)
for bar, val in zip(bars, accs):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8)
legend = [plt.Rectangle((0,0),1,1,color="#2ecc71"),
          plt.Rectangle((0,0),1,1,color="#3498db")]
ax.legend(legend, ["Neural Network", "Traditional ML"], fontsize=10)
plt.tight_layout()
plt.savefig("comparison_accuracy.png", dpi=150)
print("\nSaved: comparison_accuracy.png")

# ── Accuracy vs Train Time scatter ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for r in results:
    kw = "CNN" in r["name"] or "MLP" in r["name"] or "Softmax" in r["name"]
    c, m = ("#2ecc71", "s") if kw else ("#3498db", "o")
    ax.scatter(r["train_time"], r["acc"], c=c, marker=m, s=100, zorder=5)
    label = r["name"].split(". ")[1] if ". " in r["name"] else r["name"]
    ax.annotate(label, (r["train_time"], r["acc"]), fontsize=8,
                textcoords="offset points", xytext=(5, 3))
ax.set_xlabel("Training Time (s)")
ax.set_ylabel("Test Accuracy")
ax.set_title("Accuracy vs Training Time", fontsize=14)
ax.legend([plt.Line2D([],[],color="#3498db",marker="o",linestyle="None"),
           plt.Line2D([],[],color="#2ecc71",marker="s",linestyle="None")],
          ["Traditional ML (15k samples)", "Neural Net (full 60k)"], fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("comparison_accuracy_vs_time.png", dpi=150)
print("Saved: comparison_accuracy_vs_time.png")

# ── Confusion matrix for best model ─────────────────────────────────
best = results[0]
cm = np.array(best["cm"])
fig, ax = plt.subplots(figsize=(9, 8))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(class_names, fontsize=8)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix — {best['name']} (acc={best['acc']:.4f})", fontsize=13)
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                fontsize=7, color="white" if cm[i, j] > cm.max()/2 else "black")
plt.tight_layout()
plt.savefig("comparison_best_cm.png", dpi=150)
print(f"Saved: comparison_best_cm.png (best: {best['name']})")

# ── Per-class accuracy comparison ───────────────────────────────────
print("\n----- PER-CLASS ACCURACY (diagonal of confusion matrix / class total) -----")
header = f"{'Class':<15s}" + "".join(f"{r['name'][:12]:>12s}" for r in results)
print(header)
print("-" * len(header))
for c in range(10):
    line = f"{class_names[c]:<15s}"
    for r in results:
        cm_arr = np.array(r["cm"])
        class_acc = cm_arr[c, c] / cm_arr[c, :].sum()
        line += f"{class_acc:>12.3f}"
    print(line)

print("\nDone.")
