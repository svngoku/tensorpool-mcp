# TensorPool MCP Server (Python)

Production-friendly MCP server that wraps the TensorPool `tp` CLI via subprocess and exposes cluster and job operations as MCP tools.

- Transport: Streamable HTTP (recommended for remote/production)
- Dependency manager: `uv`
- Safe SSH key handling: validate public keys, write to a temp file, pass `-i` to `tp cluster create`
- Clear error reporting: return `stdout` on success; on error include exit code + `stdout`/`stderr`

## Quick start

Prerequisites
- Install `uv`: https://docs.astral.sh/uv/getting-started/installation/
- A TensorPool API key with access to clusters/jobs

Install and run locally (port 3000 by default):

```bash
uv python install
uv sync --locked

# Option A: .env (local only)
# Create .env with your key — DO NOT COMMIT THIS FILE
# TENSORPOOL_API_KEY=...  (or TENSORPOOL_KEY=...)

# Option B: export at runtime
TENSORPOOL_API_KEY=... uv run main.py

# Start (uses streamable-http). URL: http://127.0.0.1:3000/mcp
uv run main.py
```

Notes
- The server auto-loads `.env` for local runs but will not override existing env vars.
- The CLI expects `TENSORPOOL_KEY`. This server bridges `TENSORPOOL_API_KEY → TENSORPOOL_KEY` automatically.
- Verify CLI: `uv run tp --version` should print a version (e.g., `0.0.6`).

## MCP Inspector

UI quick start:
```bash
npx @modelcontextprotocol/inspector
```

Connect with:
- Transport: Streamable HTTP
- URL: `http://127.0.0.1:3000/mcp`
- Increase per-request timeout to 120–300s for long operations (cluster create/destroy, job push).

## Exposed tools

Clusters
- `cluster_create(ssh_public_key, instance_type, num_nodes=1, name=None)`
  - Creates a cluster. Provide an OpenSSH public key line (e.g. `ssh-ed25519 AAAA... user@host`).
- `cluster_list(org=False)`
  - Lists clusters for your account/org.
- `cluster_info(cluster_id)`
  - Shows details for a specific cluster.
- `cluster_destroy(cluster_id, confirm=False, wait=False)`
  - Destroys a cluster. Requires `confirm=true`. Use `wait=true` to block until fully deleted; otherwise returns quickly and you can poll with `cluster_info`.

Jobs
- `job_write_config(workdir, instance_type, commands, outputs=[], ignore=[], filename="tp.config.toml")`
  - Writes a minimal `tp.config.toml` next to your project. Example:
    ```toml
    instance_type = "1xH100"
    commands = [
      "pip install -r requirements.txt",
      "python train.py",
    ]
    outputs = ["checkpoints/"]
    ignore = [".venv"]
    ```
- `job_push(config_path, workdir=None)`
  - Submits a job defined by `tp.config.toml`.
- `job_list(org=False)`
  - Lists jobs.
- `job_info(job_id)`
  - Shows job details.
- `job_pull(job_id, force=False)`
  - Downloads job outputs. Use `force=true` to overwrite.
- `job_cancel(job_id, confirm=False)`
  - Cancels a running job. Requires `confirm=true`.

### Example Inspector payloads

- Destroy without blocking:
  ```json
  {
    "method": "tools/call",
    "params": {
      "name": "cluster_destroy",
      "arguments": {"cluster_id": "c-123", "confirm": true, "wait": false}
    }
  }
  ```

- Create a config and push a job:
  ```json
  {"method":"tools/call","params":{"name":"job_write_config","arguments":{
    "workdir":".",
    "instance_type":"1xH100",
    "commands":["pip install -r requirements.txt","python train.py"],
    "outputs":["checkpoints/"],
    "ignore":[".venv"],
    "filename":"tp.config.toml"
  }}}

  {"method":"tools/call","params":{"name":"job_push","arguments":{
    "config_path":"tp.config.toml"
  }}}
  ```

## Environment variables

- `TENSORPOOL_KEY` (preferred by the CLI) or `TENSORPOOL_API_KEY` (bridged)
- For local only, `.env` is auto-loaded (non-overriding). In production, set env vars in your platform (Render, Fly.io, etc.).
- Do not commit `.env`. It is ignored by `.gitignore`.

## Timeouts and interactivity

- Long operations can exceed default client timeouts. In MCP Inspector, set the per-request timeout to at least 120s (300s recommended for destructive ops).
- The server passes `--no-input` to `tp cluster destroy` to avoid interactive prompts. For other commands, ensure your API key is set so `tp` does not prompt.

## Troubleshooting

- `ERROR: TENSORPOOL_API_KEY (or TENSORPOOL_KEY) is not set in the environment.`
  - Export `TENSORPOOL_API_KEY` or `TENSORPOOL_KEY`, or create a local `.env`.
- `ERROR: 'tp' CLI not found.`
  - Ensure `tensorpool` is installed (included via `uv add tensorpool`). Verify with `uv run tp --version`.
- `MCP error -32001: Request timed out`
  - Increase client timeout; avoid long blocking operations; for `cluster_destroy` prefer `wait=false` and poll with `cluster_info`.
- Directly importing and calling tools in Python
  - When bypassing MCP (e.g., `from main import job_write_config`), pass all arguments explicitly—defaults defined with `Field(...)` are metadata, not runtime defaults.

## Deployment

- Server binds to `$PORT` (default 3000) and serves Streamable HTTP at `/mcp`.
- Render example is included (`render.yaml`, `Procfile`). Be sure to set `TENSORPOOL_KEY` (or `TENSORPOOL_API_KEY`) in the service’s environment.

## Recommendations (optional enhancements)

These are non-breaking ideas you can adopt as needed:
- Add a simple `account_me` tool (`tp me`) to validate credentials quickly.
- Return JSON when available (e.g., add `-o json` if/when the CLI supports it) and fall back to text on errors.
- Add streaming log support for long-running jobs (e.g., `tp job listen`) when client UX requires it.
- Add auth to the MCP endpoint (e.g., bearer token header) before exposing publicly.
