# Transformers

The transformer is a neural network architecture introduced in 2017 in the paper
"Attention Is All You Need". It replaced recurrence with self-attention, which lets
every token in a sequence attend to every other token in parallel.

Self-attention computes queries, keys, and values for each token. The attention
weight between two tokens is the softmax of the scaled dot product of a query and a
key. Multi-head attention runs several of these in parallel so the model can track
different kinds of relationships at once.

Large language models (LLMs) are transformers trained on large text corpora to
predict the next token. Their context window limits how much text they can read at
once, which is one reason retrieval is useful.
