# Auto-generated app.py for Hugging Face Space (inference-only)
import torch
import gradio as gr
import tiktoken
from model import GPTModel

# Recreate minimal config from the notebook
CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# merge BASE_CONFIG with chosen model details
CFG = dict(BASE_CONFIG)
CFG.update(model_configs[CHOOSE_MODEL])
# Ensure keys expected by GPTModel exist (matching training)
CFG["emb_dim"] = CFG["emb_dim"]
CFG["n_layers"] = CFG["n_layers"]
CFG["n_heads"] = CFG["n_heads"]

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# instantiate model and load weights
model = GPTModel(CFG)
state_dict = torch.load("classifier_weights_biases.pth", map_location=device)
# If the saved file is a full state_dict, load directly
if isinstance(state_dict, dict) and any(k.startswith("tok_emb") or k.startswith("pos_emb") for k in state_dict.keys()):
    model.load_state_dict(state_dict)
else:
    # Otherwise, assume it's the raw params dict; use load_weights_into_gpt if available
    try:
        from model import load_weights_into_gpt
        load_weights_into_gpt(model, state_dict)
    except Exception as e:
        raise RuntimeError("Unable to load the provided weights: " + str(e))

model.to(device)
model.eval()

# tokenizer (uses tiktoken as in notebook)
tokenizer = tiktoken.get_encoding("gpt2")

# classify_review function (copied from the notebook)
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
    # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"

# Gradio interface
def predict(text):
    return classify_review(text, model, tokenizer, device, max_length=CFG["context_length"], pad_token_id=50256)

iface = gr.Interface(fn=predict,
                     inputs=gr.Textbox(lines=4, placeholder="Enter SMS message here..."),
                     outputs="text",
                     title="SMS Spam Classifier (GPT-2 custom)",
                     description="Inference-only Gradio app using your custom GPT-2 implementation and trained weights.")

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
