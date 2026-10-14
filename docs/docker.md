# Docker

Two images are provided: a CPU image (`Dockerfile`) and a GPU image (`Dockerfile.gpu`).
Both carry the same code, so either can serve the HTTP API or run the CLI.

CPU is the supported deployment.

## CPU service

```bash
docker compose up --build -d cpu
curl http://localhost:8000/healthz
```

```json
{"status": "ok"}
```

The container listens on 8000. To publish a different host port:

```bash
TUZKAOCR_CPU_PORT=18080 docker compose up --build -d cpu
```


## GPU service

```bash
docker compose --profile gpu up --build -d gpu
```

Port 8001 by default (`TUZKAOCR_GPU_PORT`). Requires the NVIDIA Container Toolkit.

## Configuration

Both services read `tuzkaocr.env` from the repository root via `env_file`, then apply
service-specific overrides. Edit `tuzkaocr.env` and restart to change settings:

```bash
docker compose up -d --force-recreate cpu
```

The Compose file pins `TUZKAOCR_RESULTS_DIR=/app/results` and
`TUZKAOCR_SPOOL_DIR=/app/spool` for both services, so setting those two in `tuzkaocr.env`
has no effect. Everything else in [Configuration](configuration.md) applies.

## Storage

| Path in container | Backed by | Holds |
|---|---|---|
| `/app/results` | bind mount `./results` | Finished OCR output, swept by age |
| `/app/spool` | named volume (`cpu_spool` / `gpu_spool`) | Uploads in flight |

Keep `./results` on persistent storage. Each service has its own spool volume; do not share
one between services.

### File ownership

Compose runs the container as `1000:1000`, so `./results` must be writable by that UID. If
your host account differs:

```bash
TUZKAOCR_RUN_UID=$(id -u) TUZKAOCR_RUN_GID=$(id -g) docker compose up -d cpu
```

## Running the CLI in a container

For a one-off batch without installing anything on the host:

```bash
docker compose run --rm --no-deps \
  -v "$PWD/input_pages:/app/input:ro" \
  cpu python cli.py /app/input \
  --batch \
  --format txt \
  --domain kramarky \
  --out-dir /app/results \
  --workers 2
```

Results land in `./results` on the host.

## Operational notes

- Health checks hit `/healthz`; shutdown waits up to 120 s for in-flight pages.
- Terminate TLS at a reverse proxy; the app speaks plain HTTP.
- Enable an API key for any non-local deployment. For per-caller keys, mount the file and
  set `TUZKAOCR_API_KEYS_FILE`:

    ```yaml
    volumes:
      - ./api_keys.yaml:/app/api_keys.yaml:ro
    ```

- For memory sizing see [Configuration](configuration.md#memory).
