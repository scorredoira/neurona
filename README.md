# Neurona

A course to understand neural networks from scratch, all the way to a
transformer that can hold a conversation, both word by word and by
diffusion. We build everything ourselves: no AI libraries, just our own
"basic PyTorch" in Go.

Each lesson: one concept, the smallest possible example with concrete numbers,
and a program that works.

## Syllabus

| # | Lesson | What you learn | Status |
|---|---|---|---|
| 0 | [Digits](00-digits/) | Neuron, weights, softmax, error, training, hidden layer, backpropagation by hand, MNIST | done |
| 1 | [Autograd](01-autograd/) | Derivatives with numbers, chain rule, automatic backpropagation | |
| 2 | Counting letters | The simplest language model (bigrams), probability, generating text | |
| 3 | Embeddings | A network that predicts the next letter; tables (tensors) and layers | |
| 4 | Training well | Weight initialization, normalization, batches, optimizers | |
| 5 | Convolutional networks | Small detectors that slide over the image; MNIST above 98% | |
| 6 | Growing cells | From a single pixel a tiny network grows an image; tear a piece off and it regrows by itself | |
| 7 | Autoencoders | Compress a digit to 2-3 numbers and rebuild it; the latent space map | |
| 8 | Diffusion | Generate new digits starting from noise | |
| 9 | Attention | Each letter decides which previous letters to look at | |
| 10 | Transformer | A mini GPT trained on a book, and looking inside: what each part focuses on | |
| 11 | Fast version | Multiply tables with Accelerate / GPU; lower-precision weights and faster generation | |
| 12 | 3D scene from photos | 50-100 phone photos of an object turned into a scene you can turn around, seeing angles you never photographed (NeRF) | |
| 13 | Tokenizer | Split text into word pieces | |
| 14 | Search by meaning | Sentences turned into numbers to find texts that say the same thing (the basis of RAG) | |
| 15 | Reinforcement learning | A network learns to play using only "good" or "bad", without examples: our own Pong (from the screen pixels) and Breakout | |
| 16 | Connect 4 | A network that learns only by playing against itself, with no examples, until it beats you (AlphaZero) | |
| 17 | Conversation | Train on conversations, fine-tune Neurona's identity, and improve answers with ratings | |
| 18 | Text diffusion | The same transformer writes its answer all at once, filling in hidden words step by step; compared with lesson 17 | |

## References

The language path of this course (lessons 1-4 and 9-10) is inspired by Andrej
Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html).
His course uses Python and PyTorch; here everything is built from scratch in Go.
His videos are a great companion after each lesson:

| Lesson | Karpathy |
|---|---|
| 1 · Autograd | [micrograd](https://github.com/karpathy/micrograd) |
| 2 · Counting letters | makemore, part 1 (bigrams) |
| 3 · Embeddings | makemore, part 2 (MLP) |
| 4 · Training well | makemore, parts 3-4 (activations, gradients, BatchNorm, backprop) |
| 10 · Transformer | Let's build GPT · [nanoGPT](https://github.com/karpathy/nanoGPT) |
| 11 · Fast version | [llm.c](https://github.com/karpathy/llm.c) (GPT-2 in plain C) |
| 13 · Tokenizer | Let's build the GPT Tokenizer |

Other references:

- [TinyStories](https://arxiv.org/abs/2305.07759): small language models that write coherent stories.
- [MNIST](https://en.wikipedia.org/wiki/MNIST_database): the handwritten digits dataset used in lessons 0, 5 and 7-8.

## Structure

```
00-digits/   lesson 0: browser app (TypeScript)
nn/          our neural network library (Go), grows with each lesson
download/    downloads the datasets into data/ (go run ./download)
data/        downloaded datasets (not in the repo)
NN-name/     each lesson: its program (go run ./NN-name)
```

## How to run

Download the datasets (they are not in the repo):

```
go run ./download
```

Lesson 0: open `00-digits/index.html` in the browser. After changing the code:

```
cd 00-digits
esbuild app.ts --bundle --outfile=app.js
```

Retrain on MNIST:

```
cd 00-digits
esbuild mnist-train.ts --bundle --platform=node --outfile=mnist-train.js && node mnist-train.js
```

Go lessons, from the project root:

```
go run ./01-autograd
```
