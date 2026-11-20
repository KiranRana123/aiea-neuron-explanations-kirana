import csv
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


with open("../nli/data/analysis/snli_1.0_dev.tok") as f:
    lines = [line.strip() for line in f]

with open("../nli/data/analysis/preds/6_snli_1.0_dev.csv") as f:
    pred_rows = list(csv.DictReader(f))

# Combining results from SNLI to form data to feed phi3
dataset = []
for i in range(len(pred_rows)):
    dataset.append({
        "premise": lines[2 * i],
        "hypothesis": lines[2 * i + 1],
        "gt": pred_rows[i]["gt"],
        "bowman_pred": pred_rows[i]["pred"],
    })

dataset = [pair for pair in dataset if pair["gt"] != "UNK"]

print(len(dataset), "dataset")
print(dataset[0])

random.seed(42)
test_pairs = random.sample(dataset, 20) 
print("Testing on ", len(test_pairs), "pairs from the sample")

model_name = "microsoft/Phi-3-mini-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="mps", dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

rules = """You are given a premise and a hypothesis. Decide how they relate:
- entailment: if the premise is true, the hypothesis must also be true.
- contradiction: if the premise is true, the hypothesis cannot be true.
- neutral: the hypothesis might or might not be true.
Answer with exactly one word: entailment, neutral, or contradiction."""

#testing on single pair
pair = test_pairs[0]
messages = [
    {"role": "system", "content": rules},
    {"role": "user", "content": f"Premise: {pair['premise']}\nHypothesis: {pair['hypothesis']}\nAnswer:"},
]
output = pipe(messages, max_new_tokens=5, return_full_text=False, do_sample=False)
answer = output[0]["generated_text"].strip()

print("Model's output:", answer)
print("True label:", pair["gt"])