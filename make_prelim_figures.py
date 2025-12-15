import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cm = np.array([[437, 26],
               [ 22,1379]])

metrics = {
    "Class": ["OK", "stress", "accuracy", "macro avg", "weighted avg"],
    "precision": [0.9521, 0.9815, 0.9742, 0.9668, 0.9742],
    "recall":    [0.9438, 0.9843, 0.9742, 0.9641, 0.9742],
    "f1-score":  [0.9479, 0.9829, 0.9742, 0.9654, 0.9742],
    "support":   [463, 1401, 1864, 1864, 1864]
}

df = pd.DataFrame(metrics)
df.to_csv("validation_metrics.csv", index=False)

fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(cm, interpolation='nearest')
ax.set_title("Validation Confusion Matrix (2 epochs)")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["OK","stress"]); ax.set_yticklabels(["OK","stress"])
for (i,j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center")
ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
fig.tight_layout()
plt.savefig("confusion_matrix_2epochs.png", dpi=200, bbox_inches="tight")
plt.close(fig)

fig2, ax2 = plt.subplots(figsize=(6,4))
x = np.arange(2); w = 0.25
prec = [0.9521, 0.9815]
rec  = [0.9438, 0.9843]
f1   = [0.9479, 0.9829]
ax2.bar(x - w, prec, w, label="Precision")
ax2.bar(x,     rec,  w, label="Recall")
ax2.bar(x + w, f1,   w, label="F1-score")
ax2.set_xticks(x); ax2.set_xticklabels(["OK","stress"])
ax2.set_ylim(0, 1.05)
ax2.set_title("Validation Metrics by Class (2 epochs)")
ax2.legend()
fig2.tight_layout()
plt.savefig("metrics_by_class_2epochs.png", dpi=200, bbox_inches="tight")
plt.close(fig2)

print("Wrote: validation_metrics.csv, confusion_matrix_2epochs.png, metrics_by_class_2epochs.png")
