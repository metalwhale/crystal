# crystal
CRySTAL: Condensed Reinforcement using Structured Training for Adaptive Learning

## 1. Overview
### 1-1. Term explanation
- Condensed Reinforcement: Fine-tuning a small model using a larger model through reinforcement learning
- Structured Training: Constraining the inputs and outputs to follow a specific format
- Adaptive Learning: Adjusting the model to focus on a specific use case instead of general use

### 1-2. Data source
- Yahoo Japan ニュース: https://news.yahoo.co.jp/rss

### 1-3. Model training
- Pretrained model: [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- Training algorithm: [GRPO trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- Fine-tuning library: [Unsloth](https://unsloth.ai/)

## 2. Run the program
1. Prepare a machine with GPU capacity (such as an AWS `g4dn.xlarge` EC2 instance).

    Note: It's possible to run the program on Google Colab, but since this repository is private, it requires additional configuration, which I'm not yet familiar with.
2. Prerequisites:

    Change to [`./crystal-ai/`](./crystal-ai/) directory:
    ```bash
    cd ./crystal-ai/
    ```
    Install packages and dependencies:
    ```bash
    sudo apt-get update -y
    sudo apt-get install -y cmake build-essential

    curl -LsSf https://astral.sh/uv/0.8.4/install.sh | sh
    source $HOME/.local/bin/env

    uv sync
    ```
    Activate the virtual environment:
    ```bash
    source ./.venv/bin/activate
    ```
    Change to [`./storage/`](./storage/) directory:
    ```bash
    cd ../storage/
    ```
3. Run:

    Generate data:
    ```bash
    uv run ../crystal-ai/main.py data
    ```
    Train the model (`nohup` keeps the program running even after exiting the terminal):
    ```bash
    # When running the program, mysterious `core.*` files are somehow created. I'm not sure what causes this and it's quite annoying.
    # Here we are temporarily disabling core dumps, but I want to take a closer look later.
    ulimit -c 0
    nohup uv run ../crystal-ai/main.py train &
    ```
    Evaluate the model, where `${SUBDIR}` refers to the subdirectory created during training:
    ```bash
    uv run ../crystal-ai/main.py evaluate --lora-dir ./train/${SUBDIR}/lora/
    ```

## 4. References
I want to express my sincere thanks to:
- [Qwen team](https://huggingface.co/Qwen), for publishing their SOTA models that we will fine-tune for various downstream tasks.
- [GRPO trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) from HuggingFace, based on [GRPO](https://huggingface.co/papers/2402.03300), an RL algorithm proposed by DeepSeek.
- [Unsloth](https://unsloth.ai/), a fast and efficient library for fine-tuning LLMs.
- And many more...
