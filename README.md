# Photo Advisor v1.0.0

Photo Advisor is an internal Python service that checks whether a vehicle listing covers 12 required photo views. The backend sends object references over Kafka, the service reads the photos from S3, and it publishes the checklist and per-photo results back to Kafka. A direct JSON-RPC endpoint and a local development pilot are also available.

- [Kafka integration contract](contracts/kafka.md)
- [Architecture](ARCHITECTURE.md)
- [Local pilot](PILOT.md)

## Run

1. Obtain the separate model archive `PhotoAdvisor-v1.0.0-model-v19.zip` from the release owner and extract it at the repository root. The model directory must contain `manifest.json` and `weights.pt`. The source checkout contains the reference [model manifest](release/model-v19.json).
2. Copy `.env.example` to `.env` and set the service token and S3 configuration. To enable Kafka, set the broker, group and topic values and set `KAFKA_ENABLED=true`.
3. Run `docker compose up -d --build advisor`. Check `/health/ready` on port 8000.

Model v19 uses the twelve-view checklist. Automatic acceptance is disabled. The service verifies the weights at startup and uses local files.

- Weights SHA-256: `780d81138cb31f8a24e8f27604f8a945fc699a73757d0de3217edd8a643eb83b`
- Model archive SHA-256: `a75000004511a2533d9bdb57c5346d073eb136889a8d494691bb45be74fd8f9f`

## Checks

Install `requirements-dev.txt`, then run `python -m pytest -q` and `ruff check app scripts tests`. Run `node --test tests/pilot_feedback.test.cjs` for the pilot interface checks. Two tests for local data and older model artifacts skip in a clean checkout.

## CI/CD и Docker Compose

Проверки, публикация Docker-образов и автоматический деплой по тегу `v*`: [настройка CI/CD](docs/ci-cd.md).
