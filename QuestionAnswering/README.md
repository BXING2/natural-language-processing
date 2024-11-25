## General
This example demonstrates funetuning LLaMA model for question answering tasks. 

## Dataset
The dataset has around 40K movie reviews with positive and negative reviews labeled as 1 and 0, respectively. For the demonstration, 1K positive reviews and 1K negative reviews are sampled for finetuning the model.

Dataset Link: https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis

## Model
The model is LLaMA-3-1B (Large Language Model Meta AI) with a language modeling head (LlamaForCausalLM). The model consists of the embedding layer, 16 decoder layers. The weights of the 15th encoder layer (0 index) are finetuned for 10 epoches, with all other model parameters frozen.

When using the original model to generate answers, the prompt takes the following format:
> "||" + {Context} + "\n" + {Question} + "\n"

When using the finetuned model to generate answers, the prompt takes the following format:
> "\<Context\>" + {Context} + "\<Question\>" + {Question} + "\<Answer\>"

The \<Context\>, \<Question\> and \<Answer\> are newly added tokens to help seperate the context, question and answer. Also, the special tokens including bos, eos, unk, and pad are specified. Thus, the embedding layer is finetuned because of the introduction of these new tokens. 


## Evaluation
<img src="figures/train_valid_loss.png" height="300" />

**Figure 1. Loss on the train and valiation dataset during training.**

| | ROUGE Score | BLEU Score |
| --- | --- | --- |
| 1-gram | 0.54* (0.30) | 0.39* (0.19) |
| 2-gram | 0.40* (0.21) | 0.28* (0.13) |
| Average | 0.47* (0.25) | 0.34* (0.16) |

**Table 1. ROUGE and BLEU scores on test dataset from greedy decoding.**

| | ROUGE Score | BLEU Score |
| --- | --- | --- |
| 1-gram | 0.56* (0.34) | 0.38* (0.22) |
| 2-gram | 0.44* (0.25) | 0.29* (0.16) |
| Average | 0.50* (0.29) | 0.34* (0.19) |

**Table 2. ROUGE and BLEU scores on test dataset from beam search decoding.**


Through finetuning, the model achieve an accuracy of 90.5% on the test dataset. On this balanced dataset, the model exhibits similar performance in detecting positive and negative reviews according to the confusion matrix in Table 2.

## Reference
1. https://huggingface.co/docs/transformers/en/model_doc/bert
2. Kenton, Jacob Devlin Ming-Wei Chang, and Lee Kristina Toutanova. "Bert: Pre-training of deep bidirectional transformers for language understanding." Proceedings of naacL-HLT. Vol. 1. 2019.
