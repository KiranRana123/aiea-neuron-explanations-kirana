import csv
from collections import Counter
import os
os.makedirs("results", exist_ok=True)

with open("../nli/data/analysis/preds/6_snli_1.0_dev.csv") as f:
    rows = list(csv.DictReader(f))
rows = [r for r in rows if r["gt"] != "UNK"]

correct = sum(r["correct"] == "True" for r in rows)
accuracy = correct / len(rows)

print(f"Bowman LSTM accuracy: {accuracy:.4f} ({correct}/{len(rows)})")
counts = Counter(r["gt"] for r in rows)
print("Label counts:", counts)

label_accuracy = {}
for label in counts:
    label_rows = [r for r in rows if r["gt"] == label]
    label_correct = sum(r["correct"] == "True" for r in label_rows)

    label_accuracy[label] = label_correct / len(label_rows)
    print(f"{label}: {label_accuracy[label]:.4f}")

results = {
    "model": "Bowman BiLSTM",
    "dataset": "SNLI dev",
    "examples": len(rows),
    "correct": correct,
    "accuracy": accuracy,
    "per_label_accuracy": label_accuracy
}

with open("results/bowman_accuracy.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f} ({correct}/{len(rows)})\n")
    for label in label_accuracy:
        f.write(f"{label}: {label_accuracy[label]:.4f}\n")