# crystal
CRySTAL: Condensed Reinforcement using Structured Training for Adaptive Learning

## 1. Term explanation
- Condensed Reinforcement: Fine-tuning a small model using a larger model through reinforcement learning
- Structured Training: Constraining the inputs and outputs to follow a specific format
- Adaptive Learning: Adjusting the model to focus on a specific use case instead of general use

## 2. Data
Use the data collected from https://github.com/metalwhale/chloria

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

<details><summary>Run the llama.cpp server (only for <a href="#322-generate-truths">generating truths</a>)</summary>

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
nohup llama-server -m ${PWD}/models/model.gguf -ngl 99 --temp 0 --port 8080 &

# Send a chat completion request (optional)
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer no-key" -d '{"messages": [{"role":"system","content":"You are Crystal, an AI assistant."},{"role":"user","content":"Hello world"}]}'
```
</details>

### 3.2. Run the program
#### 3.2.1. Install packages
```bash
uv sync
```

#### 3.2.2. Generate truths
```bash
nohup uv run truth.py &
```
