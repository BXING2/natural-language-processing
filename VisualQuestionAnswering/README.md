## General
This example demonstrates funetuning BLIP model for visual question answering multimodal tasks. 

## Dataset
The original dataset consists of a training split containing around 10K question answer pairs, and a test split containing 3K question answer pairs. Each question answer pair is related to one image. In this example, 1K instances from the training split are sampled for finetuning the model, of which 80% is used for training and 20% is used for validation. 1K instances from the test split are sampled for testing the model performance.

Dataset Link: https://www.kaggle.com/datasets/bhavikardeshna/visual-question-answering-computer-vision-nlp

## Model
The model is BLIP (Bootstrapping Language-Image Pre-training) with a question answering head (BlipForQuestionAnswering) from Hugging Face. It consists of both transformer encoders and decoders. The encoders include a vision encoder which encodes the image and a text encoder which encodes the question together with the image. The decoder generates the answer according to the image and question. The vision encoder, text encoder and text decoder all consists of 12 BERT model encoder blocks. Notably, for the text decoder, the bidirectional self-attention is replaced with the casual self-attention for the purpose of language modeling. The weights of the classification layer in the decoder model is finetuned for 20 epoches, with all other model parameters frozen.

## Evaluation
<img src="figures/train_valid_loss.png" height="300" />

**Figure 1. Loss on the train and valiation dataset during training.**

<img src="figures/recall.png" height="300" />

**Figure 1. Recall for various number of predictions for each instance on the test dataset.**

For each question, multiple answers are generated through the beam search decoding algorithms. The recall@N here indicates the recall when each question is associated with N candidate answers. As N increaes, the recall increases since it becomes likely to generate the target answer when we give more trials. Using recall@10 as an example, after finetuning, the model improves from 0.27 to 0.49 if we generate 10 answers for each question. 

## Decoding
This section shows two examples of generating answer for the given question based on the image. The decoding algorithm used is beam search (number of beams=50). For each question, the top 10 most possible answer are kept.

<img src="figures/image986.png" height="300" />

> Quesion: "how many chairs are there" <br/>
> Target Answer: "1" <br/>

> Generated Answers Before Finetuning: "2", "two", "one", $${\color{red}"1"}$$, "0", "3", "there is no table in picture", "there are two", "4", "there are 2" <br/>

> Generated Answers After Finetuning: "2", $${\color{red}"1"}$$, "3", "4", "2 decorative vases", "2", "2 ornamental vases", "2 decorative vases 2 decorative vases", "2 decorative vases 3 decorative vases", "2 vases"

<img src="figures/image283.png" height="300" />

> Quesion: "what is right of sofa" <br/>
> Target Answer: "lamp" <br/>

> Generated Answers Before Finetuning: "table is made of wood", "tall and narrow", "it is made of wood", "table is made out of wood", "lamppost", "tall wooden pole", "it is made out of wood", "lampshade", "table is made of metal", "tall wooden fence" <br/>

> Generated Answers After Finetuning: "table", $${\color{red}"lamp"}$$, "lamppost", "decorative item", "table in front of window", "table runner", "table in front of person", "decorative item in person ' s hand", "table in front of curtain", "table lamp"


The corrected generated answer are emphasized through the red color. In the first example, the 10 candidate answer includes the target answer for both models before and after finetuning. In the second example, the finetuned model generates the correct answer while the original fails to.

## Reference
1. https://huggingface.co/docs/transformers/en/model_doc/blip
2. Li, Junnan, et al. "Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation." International conference on machine learning. PMLR, 2022.
