import string
import numpy as np

import evaluate

# --- load answers --- #
answers = np.load(
    "answers.npy",
    allow_pickle=True,
).item()

targets, generations = answers["target"], answers["generation"]

# --- load metric --- #
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")


# --- evaluate model --- #

n_answers = 1
remove_punc = True
rouge_scores, bleu_scores = [], []

for i in range(n_answers):
    
    # collect generated answers
    curr_generations = [sentences[i] for sentences in generations]

    # remove punctuations
    if remove_punc:
        targets = [sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in targets]
        curr_generations = [sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in curr_generations]

    # compute rouge and bleu score
    curr_rouge = rouge.compute(
        predictions=curr_generations,
        references=targets,
        max_order=2,
    )

    curr_bleu = bleu.compute(
        predictions=curr_generations,
        references=targets,
        max_order=2, 
    )
    
    # collect rouge/bleu scores
    rouge_scores.append([curr_rouge["rouge1"], curr_rouge["rouge2"], 0.5 * (curr_rouge["rouge1"] + curr_rouge["rouge2"])])
    curr_bleu_score = curr_bleu["precisions"]
    curr_bleu_score.append(0.5*np.sum(curr_bleu_score))
    bleu_scores.append(curr_bleu_score)

rouge_scores = np.array(rouge_scores)
bleu_scores = np.array(bleu_scores)

print(rouge_scores, bleu_scores)

rouge_ave, rouge_std = np.mean(rouge_scores, axis=0), np.std(rouge_scores, axis=0)
bleu_ave, bleu_std = np.mean(bleu_scores, axis=0), np.std(bleu_scores, axis=0)

print("rouge score:", rouge_ave, rouge_std)
print("bleu score:", bleu_ave, bleu_std)
