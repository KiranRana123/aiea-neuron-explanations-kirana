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
# questions_type1 = [
#     "Can you summarize the Diwali festival in India?",
#     "Tell me about the history of the Olympic Games.",
#     "What do you think about artificial intelligence?",
# ]
# for question in questions_type1:
#     messages = [{"role": "user", "content": question}]
#     output = pipe(messages, max_new_tokens=500, return_full_text=False, do_sample=False)
#     answer = output[0]["generated_text"]
#     print("Question:", question)
#     print("Answer:", answer)
#     print("--" * 50)

# # # Question type 2: Questions related to reasoning with 1 word answer
# questions_type2 = [
#     "Odometer is to mileage as compass is to what?",
#     """All the trees in the park are flowering trees.
#         Some of the trees in the park are dogwoods.
#         All dogwoods in the park are flowering trees.
#         If the first two statements are true, is the third statement True or False?""",
#     """Four defensive football players are chasing the opposing wide receiver, who has the ball.
#         Calvin is directly behind the ball carrier. Jenkins and Burton are side by side behind Calvin.
#         Zeller is behind Jenkins and Burton. Calvin tries for the tackle but misses and falls. Burton trips.
#         Which defensive player tackles the receiver?""",
# ]
# for question in questions_type2:
#     messages = [
#         {"role": "system", "content": "Answer with exactly one word."},
#         {"role": "user", "content": question},
#     ]
#     output = pipe(messages, max_new_tokens=5, return_full_text=False, do_sample=False)
#     answer = output[0]["generated_text"].strip()
#     print("Question:", question)
#     print("Answer:", answer)
#     print("--" * 50)

# Question type 3: Questions related to reasoning with rules and examples
questions_type3 = [
    [
        {"role": "system", "content": "Each number series follows a repeating pattern of additions "
                                      "and subtractions. Find the pattern, explain it briefly, then give "
                                      "the next number."},
        {"role": "user", "content": "2, 5, 3, 6, 4, 7, ... What number should come next?"},
        {"role": "assistant", "content": "The pattern is +3, -2, repeating. 7 - 2 = 5. Answer: 5"},
        {"role": "user", "content": "20, 16, 18, 14, 16, 12, ... What number should come next?"},
        {"role": "assistant", "content": "The pattern is -4, +2, repeating. 12 + 2 = 14. Answer: 14"},
        {"role": "user", "content": "7, 10, 8, 11, 9, 12, ... What number should come next?"},
    ],
    [
        {"role": "system", "content": "Each item is a group of three letters. Each position in the group "
                                      "follows its own alphabet pattern (for example, moving forward 1 or 2 "
                                      "letters each step). Find the pattern for each position, explain it "
                                      "briefly, then fill in the blank."},
        {"role": "user", "content": "ABC, BCD, CDE, ____, EFG"},
        {"role": "assistant", "content": "First letters: A, B, C, D, E (+1). Second letters: B, C, D, E, F (+1). "
                                         "Third letters: C, D, E, F, G (+1). Answer: DEF"},
        {"role": "user", "content": "AMB, BOC, CQD, ____, EUF"},
        {"role": "assistant", "content": "First letters: A, B, C, D, E (+1). Second letters: M, O, Q, S, U (+2). "
                                         "Third letters: B, C, D, E, F (+1). Answer: DSE"},
        {"role": "user", "content": "SCD, TEF, UGH, ____, WKL"},
    ],
    [
        {"role": "system", "content": "Each item is a group of three letters. Each position in the group "
                                        "follows its own alphabet pattern (for example, moving forward 1 or 2 "
                                        "letters each step). Find the pattern for each position, explain it "
                                        "briefly, then fill in the blank."},
        {"role": "user", "content": "ABC, BCD, CDE, ____, EFG"},
        {"role": "assistant", "content": "First letters: A, B, C, D, E i.e. (+1). Second letters: B, C, D, E, F  i.e. (+1). "
                                            "Third letters: C, D, E, F, G i.e. (+1). Answer: DEF"},
        {"role": "user", "content": "AMB, BOC, CQD, ____, EUF"},
        {"role": "assistant", "content": "First letters: A, B, C, D, E i.e. (+1). Second letters: M, O, Q, S, U i.e. (+2). "
                                            "Third letters: B, C, D, E, F i.e. (+1). Answer: DSE"},
        {"role": "user", "content": "SCD, TEF, UGH, ____, WKL"},
    ],
    [
        {"role": "system", "content": "Each number series follows a repeating group of steps. Find the pattern, "
                                      "explain it briefly, then give the next two numbers."},
        {"role": "user", "content": "50 47 44 40 37 34 30 ... Which pair of numbers comes next?"},
        {"role": "assistant", "content": "The pattern is -3, -3, -4, repeating. 30 - 3 = 27, 27 - 3 = 24. Answer: 27, 24"},
        {"role": "user", "content": "10 12 14 17 19 21 24 ... Which pair of numbers comes next?"},
        {"role": "assistant", "content": "The pattern is +2, +2, +3, repeating. 24 + 2 = 26, 26 + 2 = 28. Answer: 26, 28"},
        {"role": "user", "content": "42 40 38 35 33 31 28 ... Which pair of numbers comes next?"},
    ],
]

for messages in questions_type3:
    output = pipe(messages, max_new_tokens=150, return_full_text=False, do_sample=False)
    answer = output[0]["generated_text"].strip()

    print("Question:", messages[-1]["content"])
    print("Answer:", answer)
    print("--" * 50)