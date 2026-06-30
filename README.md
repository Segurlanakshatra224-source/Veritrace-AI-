#  VeriTrace AI

### Explainable Multimodal Visual Question Answering using Qwen2-VL

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Qwen2-VL](https://img.shields.io/badge/Model-Qwen2--VL--2B-success)
![License](https://img.shields.io/badge/License-MIT-green)

---

##  About the Project

VeriTrace AI is an **Explainable Artificial Intelligence  system built using **Qwen2-VL-2B-Instruct**. It enables users to upload an image, ask natural language questions about it, and receive not only  answer but also a explanation of the reasoning behind the prediction. It also gives the visual evidence through which it 
came to the conclusion.

Unlike traditional computer vision models that only generate predictions, VeriTrace AI focuses on **transparency** by explaining *why* the model reached its conclusion.

---

##  Project Goals

* Develop an intelligent Visual Question Answering (VQA) system.
* Improve trust in AI through explainable responses.
* Demonstrate multimodal reasoning using image and text.
* Build an entirely open-source solution without external APIs.

---

##  Features

*  Image Understanding
*  Visual Question Answering (VQA)
*  Automatic Reasoning & Explanation
*  Qwen2-VL Vision Language Model
*  Hugging Face Transformers
*  Google Colab 
*  100% Open Source

---

##  Technology Stack

| Category                | Technology                |
| ----------------------- | ------------------------- |
| Programming Language    | Python                    |
| Deep Learning Framework | PyTorch                   |
| Vision Language Model   | Qwen2-VL-2B-Instruct      |
| Model Library           | Hugging Face Transformers |
| Image Processing        | Pillow                    |
| Development Platform    | Google Colab              |

---



##  Repository Structure

```
VeriTrace-AI
│
├── README.md
├── requirements.txt
├── VeriTrace_Qwen2VL.ipynb
└── docs/
```

---

##  Installation

```bash
pip install -q  transformers accelerate torch torchvision pillow qwen-vl-utils
```

---

##  How to Run

1. Clone the repository.
2. Open the notebook in Google Colab.
3. Install the required libraries.
4. Upload an image.
5. Enter your question.
6. Receive:
   *  Answer
   *  Explanation

---

##  Sample Interaction

### Input

**Image:** Dog playing in a park

**Question**

```
Is this a dog or a cat?
```

### Output

```
Answer:
Dog

Explanation:
The model identified floppy ears, a long snout, visible paws, and body proportions that are characteristic of a dog.
```

---

##  Future Enhancements

*  Voice-based Question Answering
*  OCR Integration
*  Document Understanding
*  Video Question Answering


---

##  Model Information

| Model     | Qwen2-VL-2B-Instruct      |
| --------- | ------------------------- |           
| Framework | Hugging Face Transformers |
| Input     | Image + Text              |
| Output    | Answer + Explanation      |

---



