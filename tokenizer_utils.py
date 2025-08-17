import torch
import tiktoken

# Load GPT-2 tokenizer 
tokenizer = tiktoken.get_encoding("gpt2")


CONTEXT_LENGTH = 256
PAD_TOKEN_ID = 50256  # GPT-2's end-of-text token (also used as padding here)

def text_to_token_ids(text, tokenizer=tokenizer):
    """Convert text to token IDs, truncating/padding to CONTEXT_LENGTH."""
    token_ids = tokenizer.encode(text)
    # Truncate if longer
    token_ids = token_ids[:CONTEXT_LENGTH]
    # Pad if shorter
    if len(token_ids) < CONTEXT_LENGTH:
        token_ids += [PAD_TOKEN_ID] * (CONTEXT_LENGTH - len(token_ids))
    return torch.tensor([token_ids], dtype=torch.long)  # shape: (1, CONTEXT_LENGTH)

def token_ids_to_text(token_ids, tokenizer=tokenizer):
    """Convert token IDs back to text."""
    if token_ids.ndim == 3:
        token_ids = token_ids.squeeze(-1)  # (batch, seq_len, 1) → (batch, seq_len)
    if token_ids.ndim == 2:
        token_ids = token_ids[0]  # assume batch size 1
    return tokenizer.decode(token_ids.tolist())
