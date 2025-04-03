## Architecture

* dmodel = 512 (output dimension)
* dff = 2048
* 6 Encoder stacks -> architecture of Encoder is known
* 6 Decoder stacks -> architecture of Decoder is known
* Attention mechanism is known
* Scaled Dot-Product Attention is described
* Positional encoding using sinusoidal function

## Training

#### Data and Batching

* Development set: newstest2013

##### English-to-German
* WMT 2014 English-to-German dataset (4.5M sentence-pairs)
* Byte-pair sentence-encoding
* Source-target vocabulary of ~37000 tokens

##### English-to-French
* WMT 2014 English-to-French dataset (36M sentence-pairs)
* 32000 word-piece vocabulary

##### Common Operations
* Batching of sentences by approximate length
* Set of sentence-pairs of ~25000 source- and target tokens per batch

#### Hardware and Schedule

* 8 NVIDIA P100 GPUs
* 0.4s per train step for parameters described in paper in *Table 3*
* 100'000 train steps for base models (12h)

* 1s per train step for big models parameters described in paper in *Table 3*
* 300'000 train steps for big models (3.5 days)

#### Optimizer

* Adam (beta1=0.9, beta2=0.98, epsilon=1e-9)
* lr = dmodel e-0.5 * min(step_num e-0.5, step_num * warmup_steps e-1.5)
* warmup_steps = 4000

#### Regularization

* Residual Dropout: Pdrop = 0.1 (0.3 for big model, except E-t-F=0.1)
  * Output of each sub-layer before added to sub-layer input and normalized
  * Sums of embeddings and positional encoding in both encoder and decoder stacks
* Label Smoothing: epsilonls = 0.1

#### Hyperparameter Tuning

* Beam search: size = 4, length penalty = 0.6

## Validation

* Base model: Single model from averaging last 5 checkpoints, written at 10-minute intervals
* Big model: Single model from averaging last 20 checkpoints
* Maximum output length during inference: input_length + 50, terminate early when possible

* 28.4 BLEU on WMT 2014 English-German translation task
* 41.8 BLEU on WMT 2014 English-Frensh translation task

https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/Transformer_translate.ipynb