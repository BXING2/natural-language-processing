# import package
import string
import numpy as np

import evaluate

# --- load data -- #
answers = np.load(
    "answers_beam50.npy",
    #"answers_ref.npy",
    allow_pickle=True,
).item()

targets = answers["target"]
target_ans = targets[:, 1]
generated_ans = answers["generation"]


# --- compute recall@N --- #
counts = np.zeros(len(generated_ans[0]))
for preds, label in zip(generated_ans, target_ans):
    for i in range(len(preds)):
        if label in preds[: i+1]:
            counts[i] += 1
counts = counts / len(generated_ans)

print(counts)
