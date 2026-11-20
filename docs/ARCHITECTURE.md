# Architecture

The application is a small decoder-only GPT-style language model. Text is split into whitespace-delimited words, represented as character sequences with an end-of-word marker, and merged using learned BPE pair rules. The tokenizer maps the resulting pieces to integer IDs.

The model adds learned token and position embeddings. Each transformer block applies pre-layer-normalized multi-head self-attention with a lower-triangular causal mask, followed by a residual feed-forward network. A linear language-model head produces logits over the vocabulary.

Training windows use `x = tokens[t:t+T]` and `y = tokens[t+1:t+T+1]`; cross-entropy therefore trains the model to predict the next token at each position. Generation repeatedly samples a next token and appends it to the context.

This implementation is intentionally compact for learning. It omits many production features such as subword normalization, packed documents, validation-based checkpoint selection, mixed precision, distributed training, and robust agronomic retrieval/grounding.
