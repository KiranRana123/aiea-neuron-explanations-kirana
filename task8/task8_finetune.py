from collections import Counter
from datasets import load_dataset

data = load_dataset("stanfordnlp/snli")
data = data.filter(lambda pair: pair["label"] != -1)
print(data)
print(data["train"][0])
print("Count of Validation label:", Counter(data["validation"]["label"]))

