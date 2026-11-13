import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model_name = "microsoft/Phi-3-mini-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="mps",
    dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# messages = [
#     {"role": "system", "content": "You are a helpful AI assistant."},
#     {"role": "user", "content": "How do I solve the equation 2x + 3 = 7?"},
# ]
# output = pipe(messages, max_new_tokens=300, return_full_text=False, do_sample=False)
# print(output[0]["generated_text"])

# Task 6
# Question type 1: Open ended questions
questions = [
    "Can you summarize the Diwali festival in India?",
    "Tell me about the history of the Olympic Games.",
    "What do you think about artificial intelligence?",
]
for question in questions:
    messages = [{"role": "user", "content": question}]
    output = pipe(messages, max_new_tokens=500, return_full_text=False, do_sample=False)
    answer = output[0]["generated_text"]
    print("Question:", question)
    print("Answer:", answer)
    print("---" * 50)

# Question type 2: Questions related to reasoning with 1 word answer
questions_type2 = [
    "Odometer is to mileage as compass is to what?",
    """All the trees in the park are flowering trees.
        Some of the trees in the park are dogwoods.
        All dogwoods in the park are flowering trees.
        If the first two statements are true, is the third statement True or False?""",
    """Four defensive football players are chasing the opposing wide receiver, who has the ball.
        Calvin is directly behind the ball carrier. Jenkins and Burton are side by side behind Calvin.
        Zeller is behind Jenkins and Burton. Calvin tries for the tackle but misses and falls. Burton trips.
        Which defensive player tackles the receiver?""",
]
for question in questions_type2:
    messages = [
        {"role": "system", "content": "Answer with exactly one word."},
        {"role": "user", "content": question},
    ]
    output = pipe(messages, max_new_tokens=5, return_full_text=False, do_sample=False)
    answer = output[0]["generated_text"].strip()
    print("Question:", question)
    print("Answer:", answer)
    print("---" * 50)