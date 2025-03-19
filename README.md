# crystal
CRySTAL: Condensed Reinforcement using Structured Training for Adaptive Learning

## Prepare the deployment
### Local environment
Change to [`deployment-local`](./deployment-local/) directory:
```bash
cd ./deployment-local/
```

Create a Docker Compose `.env` file and populate its environment variables with the appropriate values:
```bash
cp local.env .env
vi .env
```

Start and get inside the container:
```bash
docker compose up --build --remove-orphans -d
docker compose exec crystal bash
```

### Remote environment
Change to [`deployment-remote`](./deployment-remote/) directory:
```bash
cd ./deployment-remote/
```

Install llama.cpp:
```bash
./install.sh
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

Change to [`crystal`](./crystal/) directory:
```bash
cd ../crystal/
```

## Run the program
Download model weights:
```bash
mkdir -p ../storage/models
curl https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/293ca9a10157b0e5fc5cb32af8b636a88bede891/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf -L -o ../storage/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
curl https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/293ca9a10157b0e5fc5cb32af8b636a88bede891/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf -L -o ../storage/models/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf
llama-gguf-split --merge ../storage/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf ../storage/models/qwen2.5-7b-instruct-q4_k_m.gguf
rm ../storage/models/qwen2.5-7b-instruct-q4_k_m-*
```

Run:
```bash
uv sync
uv run main.py
```

## Data
Use the data collected from https://github.com/metalwhale/chloria
