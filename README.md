# crystal
CRySTAL: Condensed Reinforcement using Structured Training for Adaptive Learning

## Term explanation
- Condensed Reinforcement: Fine-tuning a small model using a larger model through reinforcement learning
- Structured Training: Constraining the inputs and outputs to follow a specific format
- Adaptive Learning: Adjusting the model to focus on a specific use case instead of general use

## Data
Use the data collected from https://github.com/metalwhale/chloria

## Deployment
### 1. Set up the environment
#### a. Local environment
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
docker compose exec crystal_ai bash
```

#### b. Remote environment
Change to [`deployment-remote`](./deployment-remote/) directory:
```bash
cd ./deployment-remote/
```

Install tools:
```bash
./install_tools.sh
```

Create a Docker Compose `.env` file and populate its environment variables with the appropriate values:
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

Download models:
```bash
cd ../storage/
mkdir -p models
curl https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/8c2fd26/Qwen2.5-7B-Instruct-Q4_K_M.gguf -L -o ${PWD}/models/model.gguf
```

Start the llama.cpp [server](https://github.com/ggml-org/llama.cpp/blob/b4927/examples/server/README.md):
```bash
llama-server -m ${PWD}/models/model.gguf -ngl 99 --temp 0 --port 8080 > /dev/null 2>&1 &

# Send a chat completion request (optional)
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer no-key" -d '{"messages": [{"role":"system","content":"You are Crystal, an AI assistant."},{"role":"user","content":"Hello world"}]}'
```

Change to [`crystal_ai`](./crystal_ai/) directory:
```bash
cd ../crystal_ai/
```

Enable uv command:
```bash
source $HOME/.local/bin/env
```

### 2. Run the program
Run:
```bash
uv sync
uv run main.py
```
