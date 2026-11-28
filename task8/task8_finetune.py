import os  
import torch  
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

use_gpu = torch.cuda.is_available()
data = load_dataset("stanfordnlp/snli")
data = data.filter(lambda x: x["label"] != -1)
train_data = data["train"].shuffle(seed=42).select(range(15000))
validation_data = data["validation"]

model_name = "Qwen/Qwen2.5-0.5B"
output_dir = "Qwen_nli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(data):
    sentences = [
        "Premise: " + p + " Hypothesis: " + h
        for p, h in zip(data["premise"], data["hypothesis"])
    ]
    return tokenizer(sentences, truncation=True, max_length=128)

train_data = train_data.map(tokenize, batched=True)
validation_data = validation_data.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
    label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
    dtype=torch.bfloat16 if use_gpu else torch.float32, 
)
model.config.pad_token_id = tokenizer.pad_token_id    

lora_config = LoraConfig(
    task_type="SEQ_CLS",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  

def calculate_accuracy(predictions):
    logits, labels = predictions
    predicted_labels = np.argmax(logits, axis=1)
    accuracy = np.mean(predicted_labels == labels)
    return {"Accuracy": accuracy}

args = TrainingArguments(
    output_dir="checkpoints",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    eval_strategy="epoch",
    logging_steps=50,
    save_strategy="no",
    report_to="none",
    seed=42,
    bf16=use_gpu,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=validation_data,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=calculate_accuracy
)

trainer.train()
results = trainer.evaluate()
print("Evaluation results:", results)

os.makedirs("results", exist_ok=True)
with open("results/qwen_finetune_accuracy.txt", "w") as f:
    f.write(f"Training pairs: {len(train_data)}\n")
    f.write(f"Validation pairs: {len(validation_data)}\n")
    f.write(f"Accuracy: {results['eval_Accuracy']:.4f}\n")

model.save_pretrained(output_dir)       
tokenizer.save_pretrained(output_dir)