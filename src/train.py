from __future__ import annotations

import logging
import random
import torch
import numpy as np
from pathlib import Path
# pyrefly: ignore [missing-import]
from omegaconf import DictConfig, OmegaConf
# pyrefly: ignore [missing-import]
import hydra
from registry import build_experiment, get_experiment


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(path: str | None) -> logging.Logger:
    logger = logging.getLogger("Trainer")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if path:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def resolve_experiment_name(cfg: DictConfig) -> str:
    """Support either `experiment: oat_temporal` or `experiment.type`."""
    experiment = cfg.get("experiment")
    if isinstance(experiment, str):
        return experiment
    if experiment is not None:
        if experiment.get("type"):
            return str(experiment["type"])
    raise ValueError(
        "No experiment selected."
    )


@hydra.main(config_path="configs", config_name="config.yaml", version_base="1.1")
def main(cfg: DictConfig) -> None:
    experiment_name = resolve_experiment_name(cfg)
    spec = get_experiment(experiment_name)

    seed = int(cfg.experiment.seed)
    seed_everything(seed)

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = str(cfg.get("training", {}).get("device", default_device))
    log_path = cfg.get("training", {}).get("checkpoint_root") + cfg.get("training", {}).get("run_name") + '/train.log'
    logger = setup_logger(log_path)

    run_dir = Path(cfg.training.checkpoint_root) / cfg.training.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)

    logger.info("=" * 80)
    logger.info("Experiment: %s", spec.name)
    logger.info("Family: %s | kind: %s", spec.family, spec.kind)
    logger.info("Run: %s | seed %d | config saved to %s", cfg.training.run_name, seed, run_dir / "config.yaml")
    logger.info("Device: %s", device)
    logger.info("=" * 80)

    built = build_experiment(experiment_name, cfg, device, logger)
    trainer_cls = built.pop("trainer_cls")
    trainer_kwargs = built.pop("trainer_kwargs")

    trainer_kwargs["logger"] = logger
    trainer = trainer_cls(**trainer_kwargs)
    trainer.train()


if __name__ == "__main__":
    main()