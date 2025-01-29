
![](assets/logo.png "")

# 🌟 **Bria foundation models stack**

<!-- [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/workflow/status/your-repo-name/CI/main)](https://github.com/your-repo-name/actions)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](#)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md) -->

<p align="center">
    <a href="https://github.com/huggingface/diffusers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/datasets.svg?color=blue">
    </a>
</p>


## 🚀 **Overview**


The **Model Training Library** is an open-source library designed to streamline and accelerate the process of training machine learning models. It provides modular, scalable, and production-ready tools for training, evaluation, and data pipelines of Bria models.

![](assets/arc.png "")


### **Key Features**
- 🔌 **Foundation models backbone**: Easily extend and tune bria foundation models traning loop code.
- 🧩 **Optimized containers**: Docker images for optimal run time and GPU utilization.
- 📈 **[WIP] Data pipelines**: Full piepilens for data preperations with recipes for captions.
- 🚀 **Cloud(WIP) / Local & GPU Ready**: Optimized for multi-GPU and distributed training on Local VM, AWS or Azure.
---

## 📦 **Installation**

To get started:

- Login to HF
```bash
pip install -U "huggingface_hub[cli]"

huggingface-cli login
```


- Request access to [/BRIA-4B-Adapt](https://huggingface.co/briaai/BRIA-4B-Adapt)



- Request access to [stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)

- Run
```bash
pip install -r requirements.txt
```

## 1️⃣ Quick Example

[bria4B_adapt/example_train.py](./examples/bria4B_adapt/example_train.py)


<!-- ## 2️⃣ Supported Models -->

## 🛡️ License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙋 Community & Support
- Join our [Discord Community](https://discord.gg/Nxe9YW9zHS) for questions and discussions.
- Open an issue on GitHub Issues for bug reports or feature requests.

##
### Made with lots of GPU's ❤️ by Bria.ai
