# Phase 4: Container Cutover and Rollback

## Container Runtime Cutover

Status: completed in repository configuration.

- `Dockerfile` now supports `ARG PYTHON_VERSION` with default `3.14`.
- `docker-compose.yaml` forwards `PYTHON_VERSION` build arg and defaults to `3.14`.

This means existing builds (without extra args) now target Python 3.14.

## Build Examples

Default build (Python 3.14):

```bash
docker build -t welearn-api:py314 .
```

Explicit rollback build (Python 3.12):

```bash
docker build --build-arg PYTHON_VERSION=3.12 -t welearn-api:py312 .
```

Compose default (Python 3.14):

```bash
docker compose up --build
```

Compose rollback run (Python 3.12):

```bash
PYTHON_VERSION=3.12 docker compose up --build
```

## Rollback Tagging Guidance

Keep both tags available during rollout window:

- `welearn-api:<sha>-py314` (default target)
- `welearn-api:<sha>-py312` (rollback target)

If using a registry push flow:

```bash
docker tag welearn-api:py314 <registry>/welearn-api:<sha>-py314
docker tag welearn-api:py312 <registry>/welearn-api:<sha>-py312
docker push <registry>/welearn-api:<sha>-py314
docker push <registry>/welearn-api:<sha>-py312
```

Operational rollback is then a deployment image-tag switch to `<sha>-py312`.
