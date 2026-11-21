import csv
import random
import torch
import os
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

random.seed(42)
test_pairs = random.sample(dataset, 1000) 
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

rules_with_examples = [
    {"role": "user", "content": "Premise: A man is playing a guitar on stage.\nHypothesis: A person is making music.\nAnswer:"},
    {"role": "assistant", "content": "entailment"},
    {"role": "user", "content": "Premise: A man is playing a guitar on stage.\nHypothesis: The man is asleep in bed.\nAnswer:"},
    {"role": "assistant", "content": "contradiction"},
    {"role": "user", "content": "Premise: A man is playing a guitar on stage.\nHypothesis: The man is a famous musician.\nAnswer:"},
    {"role": "assistant", "content": "neutral"},
]

def get_label(answer):
    answer = answer.lower()
    for label in ["entailment", "neutral", "contradiction"]:
        if label in answer:
            return label
    return "invalid"

os.makedirs("results", exist_ok=True)
results_file = open("results/phi3_accuracy.txt", "w")

for strategy in ["Rules with no example", "Rules with example"]:
    phi3_preds = []
    for pair in test_pairs:
        messages = [{"role": "system", "content": rules}]
        if strategy == "Rules with example":
            messages += rules_with_examples
        messages.append({"role": "user", "content": f"Premise: {pair['premise']}\nHypothesis: {pair['hypothesis']}\nAnswer:"})

        output = pipe(messages, max_new_tokens=5, return_full_text=False, do_sample=False)
        answer = output[0]["generated_text"].strip()
        phi3_preds.append(get_label(answer))

    phi3_correct = sum(pred == pair["gt"] for pred, pair in zip(phi3_preds, test_pairs))

    print("Prompting Strategy:", strategy)
    print("  LLM accuracy:   ", phi3_correct / len(test_pairs))
    print("  Invalid answers:", phi3_preds.count("invalid"))
    results_file.write(f"{strategy}: accuracy {phi3_correct / len(test_pairs):.4f} "
                       f"({phi3_correct}/{len(test_pairs)}), invalid answers {phi3_preds.count('invalid')}\n")
    
bowman_correct = sum(pair["bowman_pred"] == pair["gt"] for pair in test_pairs)
print("Bowman accuracy:", bowman_correct / len(test_pairs))
results_file.write(f"Bowman accuracy on the same pairs: {bowman_correct / len(test_pairs):.4f}\n")
results_file.close()