## General
This example demonstrates funetuning LLaMA model for question answering tasks. 

## Dataset
The original dataset has around 10K question answer pairs from financial reports. Each instance includes a question, answer, and context from which the answer is extracted. In this example, 1K instances are sampled for finetuning the model. The dataset is splited into three parts, 60% for training, 20% for validation and 20% for testing.

Dataset Link: https://www.kaggle.com/datasets/yousefsaeedian/financial-q-and-a-10k

## Model
The model is LLaMA-3-1B (Large Language Model Meta AI) with a language modeling head (LlamaForCausalLM) from Hugging Face. The model consists of the embedding layer, 16 decoder layers. The weights of the 15th decoder layer (index starting from 0) are finetuned for 10 epoches, with all other model parameters frozen.

When using the original model to generate answers as benchmarks, the prompt takes the following format:
> "\<|begin_of_text|\>" + {Context} + "\n" + {Question} + "\n"

Here, the "\<|begin_of_text|\>" represents the start of the sentence token, and "\n" represents the new line which is used to seperate the context, question and answer. {Context} and {Question} indicates the content of the context and question.

When using the finetuned model to generate answers, the prompt takes the following format:
> "\<Context\>" + {Context} + "\<Question\>" + {Question} + "\<Answer\>"

Here, the "\<Context\>", "\<Question\>" and "\<Answer\>" are newly added tokens to help indicate the start of the context, question and answer. Meanwhile, the special tokens including start of sentence, end of sentence, unknown, and padding tokens are specified. As a result, the embedding layer is finetuned because of the introduction of these new tokens. Similarly, {Context} and {Question} indicates the content of the context and question.

## Evaluation
<img src="figures/train_valid_loss.png" height="300" />

**Figure 1. Loss on the train and valiation dataset during training.**

<img src="figures/recall.png" height="300" />

**Figure 1. Recall for various number of predictions for each instance on the test dataset.**

Through finetuning, the model improves from 0.29 to 0.50 on ROUGE score, and from 0.19 to 0.34 on BLEU score, respectively, on the test dataset via beam search. Moreover, compared with greedy decoding, the beam search helps improve the decoding quality slightly. For example, the ROUGE score improves from 0.47 for greedy decoding to 0.50 for beam search. 

## Decoding
This section shows two examples of generating answer for the given question based on the image. 

<img src="figures/train_valid_loss.png" height="300" />

> Quesion: "how many chairs are there" <br/>
> Target Answer: "1" <br/>
> Generated Answers Before Finetuning: "2", "two", "one", $${\color{red}"1"}$$, "0", "3", "there is no table in picture", "there are two", "4", "there are 2" <br/>
> Generated Answers After Finetuning: "2", $${\color{red}"1"}$$, "3", "4", "2 decorative vases", "2", "2 ornamental vases", "2 decorative vases 2 decorative vases", "2 decorative vases 3 decorative vases", "2 vases"

<img src="figures/train_valid_loss.png" height="300" />

> Quesion: "what is right of sofa" <br/>
> Target Answer: "lamp" <br/>
> Generated Answers Before Finetuning: "table is made of wood", "tall and narrow", "it is made of wood", "table is made out of wood", "lamppost", "tall wooden pole", "it is made out of wood", "lampshade", "table is made of metal", "tall wooden fence" <br/>
> Generated Answers After Finetuning: "table", "lamp", "lamppost", "decorative item", "table in front of window", "table runner", "table in front of person", "decorative item in person ' s hand", "table in front of curtain", "table lamp"


Here, the beam search generates a good answer while the greedy search fails to answers the question correctly.

## Reference
1. https://huggingface.co/docs/transformers/en/model_doc/llama3
2. Touvron, Hugo, et al. "Llama: Open and efficient foundation language models." arXiv preprint arXiv:2302.13971 (2023).
