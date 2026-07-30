#  VeriTrace AI

### Explainable Multimodal Visual Question Answering using Qwen2-VL

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Qwen2-VL](https://img.shields.io/badge/Model-Qwen2--VL--2B-success)
![License](https://img.shields.io/badge/License-MIT-green)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991)

---
 

##  About the Project

VeriTrace AI is an **Explainable Artificial Intelligence  system built using **Qwen2-VL-2B-Instruct**. The system accepts multiple input modalities—including text, images, audio, and video—and generates intelligent responses while providing insights into the model's reasoning process.  It enables user to ask natural language questions about it, and receive not only  answer but also a explanation of the reasoning behind the prediction. It also gives the  evidence through which it came to the conclusion.

Unlike traditional computer vision models that only generate predictions, VeriTrace AI focuses on **transparency** by explaining *why* the model reached its conclusion.

---
## Features
* Text based question answering
* Image understanding and reasoning
* speech to text using
* video analysis
* mudimodal reasoning with Qwen
* Hugging Face Transformers
* 100% Open Source
  

---

##  Project Goals
 - multimodal input processor
 - Accurate AI Reasoning
 - Speech Understanding
 - Explainable AI
 - Response Verification

---

---

##  Technology Stack

| Category                | Technology                |
| ----------------------- | ------------------------- |
| Programming Language    | Python                    |
| Deep Learning Framework | PyTorch                   |
| Vision Language Model   | Qwen2-VL-2B-Instruct      |
| Audio Language Model    | Qwen2-VL-7B-Instruct      |
| Model Library           | Hugging Face Transformers |
| Image Processing        | Pillow                    |
| Development Platform    | VS Code                   |

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
## Libraries
* Python
* Transformers
* Librosa
* Pillow
* Numpy
* Open CV
* Captum

---


##  How to Run



---
## Explainability
1. Confidence scores
2. Attention visualization
3. Saliency maps
4. Token importance
5. Reasoning interpretation

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



---

##  Model Information

| Model        | Qwen2-VL-2B-Instruct         |
| ---------    | -------------------------    |           
| Framework    | Hugging Face Transformers    |
| Input        | Image / Text / audio / video |
| Output       | Answer + Explanation         |

---



