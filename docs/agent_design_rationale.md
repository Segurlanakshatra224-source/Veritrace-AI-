# Agent Design Rationale

## Objective

VeriTrace AI is designed to answer questions about images while explaining its reasoning.

---

## Why Qwen2-VL?

Qwen2-VL was selected because it supports multimodal reasoning by processing both images and text simultaneously. It produces detailed answers and explanations without requiring external APIs.
Its opensource , supported by hugging face and googlecolab.

---

## Why Hugging Face?

Provides pretrained models that are easy to use and reproduce.

---

## Why Google Colab?

Offers free GPU support for running large vision-language models , easy to use and understand.

---

## Design Workflow

Image
↓

Question

↓

Processor

↓

Vision Encoder

↓

Language Encoder

↓

Fusion

↓

Reasoning

↓

Answer

↓

Explanation
