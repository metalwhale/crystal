# crystal
CRySTAL: Condensed Reinforcement using Structured Training for Adaptive Learning

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
docker compose exec crystal bash
```

Run:
```bash
uv sync
uv run main.py
```

## Data
Use the data collected from https://github.com/metalwhale/chloria
