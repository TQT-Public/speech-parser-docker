docker build --pull --rm -f "Dockerfile" -t speech-parser-gpu:latest "."
docker compose up -d