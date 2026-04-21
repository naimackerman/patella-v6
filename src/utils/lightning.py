"""Shared PyTorch Lightning logger helpers."""

from __future__ import annotations

import os
from pathlib import Path

from pytorch_lightning.loggers import CSVLogger

from src.utils.env import load_repo_env


def _resolve_wandb_api_key(wandb_cfg) -> str:
    """Resolve the W&B API key from env first, then config fallback."""
    env_key = os.environ.get("WANDB_API_KEY", "").strip()
    if env_key:
        return env_key

    for attr_name in ("api_key", "WANDB_API_KEY"):
        cfg_value = getattr(wandb_cfg, attr_name, "")
        if cfg_value is None:
            continue
        cfg_key = str(cfg_value).strip()
        if cfg_key:
            os.environ["WANDB_API_KEY"] = cfg_key
            return cfg_key

    return ""


def _resolve_wandb_run_id(wandb_cfg) -> str:
    env_value = os.environ.get("WANDB_RUN_ID", "").strip()
    if env_value:
        return env_value

    cfg_value = getattr(wandb_cfg, "id", "")
    if cfg_value is None:
        return ""
    return str(cfg_value).strip()


def _resolve_wandb_resume_mode(wandb_cfg, run_id: str) -> str:
    env_value = os.environ.get("WANDB_RESUME", "").strip()
    if env_value:
        return env_value

    cfg_value = getattr(wandb_cfg, "resume", "")
    cfg_text = "" if cfg_value is None else str(cfg_value).strip()
    if cfg_text:
        return cfg_text

    if run_id:
        return "allow"
    return ""


def build_loggers(cfg, run_name: str):
    """Build CSV and optional W&B loggers from config."""
    load_repo_env(getattr(cfg, "project_root", Path(__file__).resolve().parents[2]))
    output_dir = Path(cfg.output_dir)
    loggers = [CSVLogger(save_dir=str(output_dir), name="lightning_logs")]

    wandb_cfg = getattr(cfg, "wandb", None)
    if wandb_cfg is None or not bool(getattr(wandb_cfg, "enabled", True)):
        return loggers if len(loggers) > 1 else loggers[0]

    mode = str(getattr(wandb_cfg, "mode", "online")).lower()
    if mode == "disabled":
        return loggers if len(loggers) > 1 else loggers[0]
    api_key = _resolve_wandb_api_key(wandb_cfg)
    if mode == "online" and not api_key:
        print("W&B is enabled in config but no WANDB_API_KEY is set; using CSV logger only.")
        return loggers if len(loggers) > 1 else loggers[0]

    try:
        from pytorch_lightning.loggers import WandbLogger
    except ImportError:
        return loggers if len(loggers) > 1 else loggers[0]

    run_id = _resolve_wandb_run_id(wandb_cfg)
    resume_mode = _resolve_wandb_resume_mode(wandb_cfg, run_id)

    loggers.insert(
        0,
        WandbLogger(
            name=run_name,
            project=getattr(wandb_cfg, "project", None),
            entity=getattr(wandb_cfg, "entity", None),
            save_dir=str(output_dir),
            offline=(mode == "offline"),
            log_model=False,
            id=run_id or None,
            resume=resume_mode or None,
        ),
    )
    return loggers if len(loggers) > 1 else loggers[0]
