## General
This example demonstrates funetuning GPT model for language modeling tasks. 

## Dataset
The dataset contains the text from the first book of the "Harry Potter" series.

Dataset Link: https://www.kaggle.com/datasets/prashantkarwasra/books-dataset-text-generation/data

## Model
The model is GPT2 (Generative Pre-Training) Small model with a language modeling head (GPT2LMHeadModel) from HuggingFace. The model consists of the embedding layer, 12 decoder layers. The weights of the 10th and 11th decoder layers (index starting from 0) are finetuned for 20 epoches, with all other model parameters frozen.

## Evaluation

<img src="figures/train_valid_loss.png" height="300" />

**Figure 1. Loss on the train and valiation dataset during training.**

| | Loss | Perplexity |
| --- | --- | --- |
| Train | 2.79 | 16.24 |
| Validation | 3.45 | 31.36 |
| Test | 3.44 | 31.26 |

**Table 1. Loss and perplexity on train/validation/test dataset.**

Through finetuning, the model achieves a perplexity of 31.26 on the test dataset.

## Decoding

In addition to next token predictions, the decoding algorithms also significantly influence the language generations. This section lists some examples generated from various decoding algorithms. In all cases, only the start of the sentence token "\<|startoftext|\>" is provided to generate a sequence of tokens. 

1. Greedy <br/>
> "I'm not sure what's going on," said Ron. "I don't know what's going on. I don't know what's going on."

2. Beam search <br/>
> "I don't know," said Hagrid. "I don't know what's going to happen. I don't know what's going to happen. I don't know what's going to happen. I don't know what's going to happen. I don't know what's going to happen. I don't know what's going to happen. I don't know what's going to happen. I don't know what's going to happen

3. Beam search with repetition penalty <br/>
> "I don't know what to say," said Hagrid. "What do you think?"

4. Top P (P=0.6) <br/>
> "I don't know what's going on here, but there's no reason why you should be allowed to do anything to the people of Flint," said Ron.

5. Top K (K=20) <br/>
> "I don't know what's going to happen tomorrow," said Potter. "There'll be a lot of stuff to learn. I haven't really gotten into magic yet... but, if you ask me about that..."

6. Top K with enhanced temperature (K=20, temperature=10) <br/>
> "Magrid said, while Harry took it to her chest as a last-minute gift he'd given Professor Quirrell."

## Reference
1. https://huggingface.co/docs/transformers/en/model_doc/gpt2
2. Radford, Alec. "Improving language understanding by generative pre-training." (2018).
3. Radford, Alec, et al. "Language models are unsupervised multitask learners." OpenAI blog 1.8 (2019): 9.
