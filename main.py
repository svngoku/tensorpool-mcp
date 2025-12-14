"""
TensorPool MCP Server (FastMCP)

Implements tools that wrap the TensorPool `tp` CLI for clusters and jobs.
- Requires TENSORPOOL_API_KEY in the environment (the CLI reads it)
- Safer SSH public key handling: accept the public key text, validate, write to a temp file, pass with `-i` to `tp cluster create`
- Returns stdout on success; on error returns a string containing exit code, stdout, and stderr for agent reasoning
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field


# Load .env if present (local/dev). Does not override existing env vars.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

mcp = FastMCP("TensorPool MCP", stateless_http=True)


def _run_tp(args: list[str], cwd: Optional[str] = None, timeout_s: int = 600) -> str:
    """Run the `tp` CLI with the provided arguments and return a text result."""
    env = os.environ.copy()

    # TensorPool CLI expects the API key in env
    if not env.get("TENSORPOOL_API_KEY"):
        return "ERROR: TENSORPOOL_API_KEY is not set in the environment."

    cmd = ["tp", *args]
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError:
        return "ERROR: 'tp' CLI not found. Install with: uv add tensorpool"
    except subprocess.TimeoutExpired:
        return f"ERROR: Command timed out after {timeout_s}s: {shlex.join(cmd)}"

    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()

    if p.returncode == 0:
        return out or "OK"
    return f"ERROR (exit={p.returncode})\nSTDOUT:\n{out}\n\nSTDERR:\n{err}"


def _validate_public_key(pub: str) -> str:
    """Minimal validation to prevent accidental private key submission."""
    pub = (pub or "").strip()

    if not pub:
        raise ValueError("ssh_public_key is empty")

    # Reject obvious private key material
    if "BEGIN OPENSSH PRIVATE KEY" in pub or "BEGIN RSA PRIVATE KEY" in pub:
        raise ValueError("ssh_public_key looks like a PRIVATE key; refusing")

    # Basic OpenSSH public key formats
    allowed_prefixes = ("ssh-ed25519 ", "ssh-rsa ", "ecdsa-sha2-nistp256 ")
    if not pub.startswith(allowed_prefixes):
        raise ValueError(
            "ssh_public_key must start with a valid OpenSSH public key prefix "
            "(e.g., ssh-ed25519, ssh-rsa, ecdsa-sha2-nistp256)"
        )

    return pub


def _write_temp_public_key(pub: str) -> str:
    """
    Write the public key to a temp file and return the path. (0600 perms)
    Caller is responsible for deleting the file.
    """
    fd, path = tempfile.mkstemp(prefix="tp_pubkey_", suffix=".pub")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(pub)
            f.write("\n")
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            os.unlink(path)
        except Exception:
            pass
        raise
    return path


# --------------------
# Cluster tools
# --------------------
@mcp.tool(
    title="Create cluster",
    description="Create a TensorPool GPU cluster using an SSH public key.",
)
def cluster_create(
    ssh_public_key: str = Field(
        description="OpenSSH public key text (single line), e.g. 'ssh-ed25519 AAAA... user@host'"
    ),
    instance_type: str = Field(
        description="Instance type, e.g. 1xH100, 8xH200, 8xB200"
    ),
    num_nodes: int = Field(
        default=1,
        description="Number of nodes (1 for single-node; >1 for multi-node types)",
    ),
    name: Optional[str] = Field(default=None, description="Optional cluster name"),
) -> str:
    try:
        pub = _validate_public_key(ssh_public_key)
    except ValueError as e:
        return f"ERROR: {e}"

    key_path = None
    try:
        key_path = _write_temp_public_key(pub)
        args = ["cluster", "create", "-i", key_path, "-t", instance_type]
        if num_nodes and num_nodes > 1:
            args += ["-n", str(num_nodes)]
        if name:
            args += ["--name", name]
        return _run_tp(args)
    finally:
        if key_path:
            try:
                os.unlink(key_path)
            except Exception:
                # Best-effort cleanup
                pass


@mcp.tool(title="List clusters", description="List your TensorPool clusters.")
def cluster_list(
    org: bool = Field(
        default=False, description="List organization clusters (if supported)"
    )
) -> str:
    args = ["cluster", "list"]
    if org:
        args += ["--org"]
    return _run_tp(args)


@mcp.tool(title="Cluster info", description="Get info for a TensorPool cluster by id.")
def cluster_info(
    cluster_id: str = Field(description="Cluster id"),
) -> str:
    return _run_tp(["cluster", "info", cluster_id])


@mcp.tool(
    title="Destroy cluster",
    description="Destroy a TensorPool cluster by id (requires confirm=true).",
)
def cluster_destroy(
    cluster_id: str = Field(description="Cluster id"),
    confirm: bool = Field(
        default=False, description="Must be true to actually destroy the cluster"
    ),
) -> str:
    if not confirm:
        return "Refusing to destroy cluster: set confirm=true to proceed."
    return _run_tp(["cluster", "destroy", cluster_id])


# --------------------
# Job tools
# --------------------
@mcp.tool(
    title="Write tp.config.toml",
    description="Generate a tp.config.toml for tp job push.",
)
def job_write_config(
    workdir: str = Field(description="Directory to write the config in"),
    instance_type: str = Field(description='e.g. "1xH100"'),
    commands: list[str] = Field(
        description='Shell commands to run sequentially, e.g. ["pip install -r requirements.txt","python train.py"]'
    ),
    outputs: list[str] = Field(
        default_factory=list,
        description='Files/dirs/globs to save, e.g. ["checkpoints/"]',
    ),
    ignore: list[str] = Field(
        default_factory=list,
        description='Paths/globs to exclude from upload, e.g. [".venv"]',
    ),
    filename: str = Field(default="tp.config.toml", description="Config filename"),
) -> str:
    wd = Path(workdir).expanduser().resolve()
    wd.mkdir(parents=True, exist_ok=True)
    cfg_path = wd / filename

    def toml_list(xs: list[str]) -> str:
        return "[\n" + "".join([f'  "{x}",\n' for x in xs]) + "]"

    toml_text = (
        f'instance_type = "{instance_type}"\n'
        f"commands = {toml_list(commands)}\n"
        f"outputs = {toml_list(outputs)}\n"
        f"ignore = {toml_list(ignore)}\n"
    )

    cfg_path.write_text(toml_text, encoding="utf-8")
    return f"Wrote {cfg_path}"


@mcp.tool(title="Push job", description="Submit a job using tp job push <config_path>.")
def job_push(
    config_path: str = Field(description="Path to tp.config.toml"),
    workdir: Optional[str] = Field(
        default=None, description="Optional working directory for the push"
    ),
) -> str:
    return _run_tp(["job", "push", config_path], cwd=workdir)


@mcp.tool(title="List jobs", description="List TensorPool jobs.")
def job_list(
    org: bool = Field(default=False, description="List organization jobs"),
) -> str:
    args = ["job", "list"]
    if org:
        args += ["--org"]
    return _run_tp(args)


@mcp.tool(title="Job info", description="Get detailed information about a job.")
def job_info(job_id: str = Field(description="Job id")) -> str:
    return _run_tp(["job", "info", job_id])


@mcp.tool(
    title="Pull job outputs", description="Download output files from a completed job."
)
def job_pull(
    job_id: str = Field(description="Job id"),
    force: bool = Field(default=False, description="Overwrite existing files"),
) -> str:
    args = ["job", "pull", job_id]
    if force:
        args += ["--force"]
    return _run_tp(args)


@mcp.tool(
    title="Cancel job", description="Cancel a running job (requires confirm=true)."
)
def job_cancel(
    job_id: str = Field(description="Job id"),
    confirm: bool = Field(
        default=False, description="Must be true to actually cancel the job"
    ),
) -> str:
    if not confirm:
        return "Refusing to cancel job: set confirm=true to proceed."
    return _run_tp(["job", "cancel", job_id])


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
