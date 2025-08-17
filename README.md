# 📧 GPT-2–Style Spam Classifier (PyTorch + Gradio)

A custom-built GPT-2 inspired transformer that classifies email text as **spam** or **not spam** — deployed with a Gradio app for instant, interactive testing.

---

## 🚀 Project Highlights
- 🧠 **Scratch-built Transformer**: Implemented Multi-Head Attention, FeedForward, and LayerNorm modules from the ground up in PyTorch.  
- 📊 **High Accuracy**: Achieved **99.13% training**, **97.32% validation**, and **95.67% test accuracy** on a balanced dataset of 1,530 emails (765 spam / 765 ham).  
- 🌐 **Interactive Demo**: Deployed with Gradio — paste any email text and instantly get predictions.  

---

## 🏗️ Model Overview
- **Architecture**: GPT-2 style decoder-only transformer (≈124M parameters)  
- **Config**: `emb_dim=768`, `n_layers=12`, `n_heads=12`, `context_length=1024`, `dropout=0.1`, `qkv_bias=True`  
- **Tokenizer**: GPT-2 (`tiktoken`)  
- **Objective**: Binary classification (spam vs not spam)  

---

## 📊 Results
- **Training Accuracy**: 99.13%  
- **Validation Accuracy**: 97.32%  
- **Test Accuracy**: 95.67%  

Plots:  
![Accuracy](accuracy-plot.pdf)  
![Loss](loss-plot.pdf)  


