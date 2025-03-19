# crystal
CRySTAL: Condensed Reinforcement using Structured Training for Adaptive Learning

## Term explanation
- Condensed Reinforcement: Fine-tuning a small model using a larger model through reinforcement learning
- Structured Training: Constraining the inputs and outputs to follow a specific format
- Adaptive Learning: Adjusting the model to focus on a specific use case instead of general use

## Data
Use the data collected from https://github.com/metalwhale/chloria

## Local development
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
docker compose exec crystal-ai bash
```

Run:
```bash
uv sync
uv run main.py
```
