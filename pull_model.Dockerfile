#syntax = docker/dockerfile:1.4

FROM ollama/ollama:latest AS ollama
FROM babashka/babashka:latest

# just using as a client - never as a server
COPY --from=ollama /bin/ollama ./bin/ollama

COPY pull_model.clj pull_model.clj

ENTRYPOINT ["bb", "-f", "pull_model.clj"]

