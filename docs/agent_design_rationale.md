# Agent Design Rationale

## Objective

VeriTrace AI is designed to answer questions about images while explaining its reasoning & the way it approached the conclusion.

---

## Why Qwen2-VL?

Qwen2-VL was selected because it supports multimodal reasoning by processing images , text , video and audio simultaneously. It produces detailed answers and explanations without requiring external APIs.
Its opensource , supported by hugging face .
Its best balance of reasoning and Explainability.

---

## Why Hugging Face?

Provides pretrained models that are easy to use and reproduce.

---

## Why Google Colab?

Offers free GPU support for running large vision-language models , easy to use and understand.

---

## Design Workflow

Users Input
[Image / Text / Audio  / Video]
↓

Question

↓

Processor

↓

Vision Encoder

↓

Language Encoder

↓

Audio Encoder 

↓

Fusion

↓

Reasoning

↓

Answer

↓

Explanation

↓

Confidence
