# crystal
CRySTAL: Condensed Reinforcement using Structured Training for Adaptive Learning

## 1. Term explanation
- Condensed Reinforcement: Fine-tuning a small model using a larger model through reinforcement learning
- Structured Training: Constraining the inputs and outputs to follow a specific format
- Adaptive Learning: Adjusting the model to focus on a specific use case instead of general use

## 2. Data
### 2.a. [Extraction](./crystal-ai/crystal/extraction/), [summarization](./crystal-ai/crystal/summarization/), [haiku](./crystal-ai/crystal/haiku/) tasks
Use the news data collected from https://github.com/metalwhale/chloria

### 2.b. [Chatbot](./crystal-ai/crystal/chatbot/) task
[Download Slack workspace data as a `.zip` file](https://slack.com/help/articles/201658943-Export-your-workspace-data) and extract its contents into `./storage/data/chatbot/origin/slack/` directory (Ref: [`data.py`](./crystal-ai/crystal/chatbot/data.py) file)

## 3. Deployment
### 3.1. Set up the environment
#### 3.1.a. Local environment
Change to [`deployment-local`](./deployment-local/) directory:
```bash
cd ./deployment-local/
```

Create a Docker Compose `.env` file and populate its environment variables with the appropriate values:
- `CHLORIA_API_KEY`, `CHLORIA_API_SECRET`: Obtained from https://chloria.wave.metalwhale.dev/api
```bash
cp local.env .env
vi .env
```

Start and get inside the container:
```bash
docker compose up --build --remove-orphans -d
docker compose exec crystal-ai bash
```

#### 3.1.b. Remote environment
Change to [`deployment-remote`](./deployment-remote/) directory:
```bash
cd ./deployment-remote/
```

Create a Docker Compose `.env` file (similar to the local environment):
```bash
cp remote.env .env
vi .env
```

Export the environment variables:
```bash
set -a
source .env
set +a
```

Install uv command:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Change to [`crystal-ai`](./crystal-ai/) directory:
```bash
cd ../crystal-ai/
```

<details><summary>Run the llama.cpp server (only for <a href="#322-generate-data">generating data</a>)</summary>

Open another terminal and change to [`deployment-remote`](./deployment-remote/) directory:
```bash
cd ../deployment-remote/
```

Install llama.cpp tools:
```bash
./install_llamacpp.sh
```

Download models:
```bash
cd ../storage/
mkdir -p models
curl https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/8c2fd26/Qwen2.5-7B-Instruct-Q4_K_M.gguf -L -o ${PWD}/models/model.gguf
```

Start the llama.cpp [server](https://github.com/ggml-org/llama.cpp/blob/b4927/examples/server/README.md):
```bash
nohup llama-server -m ${PWD}/models/model.gguf -ngl 99 --temp 0 --port 8080 &> nohup_llamacpp.out &

# Send a chat completion request (optional)
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer no-key" -d '{"messages": [{"role":"system","content":"You are Crystal, an AI assistant."},{"role":"user","content":"Hello world"}]}'
```
</details>

### 3.2. Run the program
#### 3.2.1. Prerequisites
Install packages:
```bash
sudo apt-get update -y
sudo apt-get install -y build-essential

uv sync
```

Activate the virtual environment:
```bash
source ./.venv/bin/activate
```

Change to [`storage`](./storage/) directory:
```bash
cd ../storage/
```

#### 3.2.2. Generate data
```bash
# Summarization task
uv run ../crystal-ai/main.py data summarization ./ 2025-08-02

# Chatbot task
uv run ../crystal-ai/main.py data chatbot ./

# Extraction task
LLAMACPP_MAX_TOKENS=4096 nohup uv run ../crystal-ai/main.py data extraction ./ 2025-08-02 &

# Haiku task
uv run ../crystal-ai/main.py data haiku ./ 2025-08-02
```

#### 3.2.3. Train models
```bash
# Summarization task
nohup uv run ../crystal-ai/main.py train summarization ./ &

# Chatbot task
nohup uv run ../crystal-ai/main.py train chatbot ./ &

# Extraction task
nohup uv run ../crystal-ai/main.py train extraction ./ &

# Haiku task
uv run python -c 'import nltk; nltk.download("cmudict")'
nohup uv run ../crystal-ai/main.py train haiku ./ &
```

#### 3.2.4. Evaluate models
```bash
# Summarization task
nohup uv run ../crystal-ai/main.py eval summarization ./ ./train/summarization/${TASK_SUBDIR}/lora &

# Chatbot task
uv run ../crystal-ai/main.py eval chatbot ./ ./train/chatbot/${TASK_SUBDIR}/lora

# Extraction task
uv run ../crystal-ai/main.py eval extraction ./ ./train/extraction/${TASK_SUBDIR}/lora

# Haiku task
uv run ../crystal-ai/main.py eval haiku ./ ./train/haiku/${TASK_SUBDIR}/lora
```
