from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Callable
from omegaconf import DictConfig # pyrefly: ignore [missing-import]
from OAT.baselines import _get

MODEL_TYPES = {
    "transformer": "transformer", "oat": "transformer",
    "tcn": "tcn", "cnn": "tcn", "temporal_cnn": "tcn",
    "global": "global", "global_attention": "global",
}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    family: str
    kind: str
    build: Callable[..., dict[str, Any]]
REGISTRY: dict[str, ExperimentSpec] = {}

def get_experiment(name: str) -> ExperimentSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown experiment '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]

def build_experiment(name: str, cfg: DictConfig, device: str, logger):
    spec = get_experiment(name)
    components = spec.build(cfg, device, logger)
    logging.getLogger("Trainer").info(
        "Built %s (%s/%s): train=%d, val=%d",
        spec.name,
        spec.family,
        spec.kind,
        components["train_size"],
        components["val_size"],
    )
    return components

def register(name: str, family: str, kind: str):
    def decorator(function: Callable[..., dict[str, Any]]):
        REGISTRY[name] = ExperimentSpec(name, family, kind, function)
        return function
    return decorator

# ==================================================================
# Registry

@register("oat_temporal", family="oat", kind="temporal")
def build_oat_temporal(cfg: DictConfig, device: str, logger):
    return _build_oat(cfg, "temporal", device, logger)

def build_temporal_model(model_cfg, num_keypoints, window_size, max_offset):
    """
    Build the window model named by model_cfg.type ("transformer" if absent).
    Every model returned takes (window, vis_mask, frame_offsets=, pad_mask=) and
    returns at least coordinates / coords_dense / occluded_mask.
    """
    kind = str(_get(model_cfg, "type", "transformer")).lower()
    if kind not in MODEL_TYPES:
        raise ValueError(f"unknown model.type {kind!r}; choose one of {sorted(MODEL_TYPES)}")
    if MODEL_TYPES[kind] == "transformer":
        from OAT.models import TemporalModel
        return TemporalModel(model_cfg, num_keypoints=num_keypoints, window_size=window_size, max_offset=max_offset)
    if MODEL_TYPES[kind] == "global":
        from OAT.models import GlobalSelfAttentionModel
        return GlobalSelfAttentionModel(model_cfg, num_keypoints=num_keypoints, window_size=window_size, max_offset=max_offset)
    from OAT.baselines import TemporalCNN
    return TemporalCNN(model_cfg, num_keypoints=num_keypoints, window_size=window_size, max_offset=max_offset)

# ==================================================================
# Mapping
def _build_oat(cfg: DictConfig, kind: str, device: str, logger):
    from OAT.dataset import RawDataset
    from OAT.masker import RecordingOcclusion
    from OAT.loss import TemporalKeypointLoss
    from OAT.trainer import Trainer
    from OAT.wrapper import TemporalWrapper
    import copy

    # Dataset
    train_cfg = cfg.train_dataset
    val_cfg = cfg.validation_dataset
    mask_cfg = cfg.masker
    val_mask_cfg = cfg.val_masker
    loss_cfg = cfg.loss
    training_cfg = cfg.training

    raw_train = RawDataset(train_cfg)
    raw_val = RawDataset(val_cfg)

    occ_train = RecordingOcclusion(raw_train.T, raw_train.N, fps=raw_train.fps, config=mask_cfg, poses=raw_train.deform_global, seed=train_cfg.seed)
    occ_val = RecordingOcclusion(raw_val.T, raw_val.N, fps=raw_val.fps, config=val_mask_cfg, poses=raw_val.deform_global, seed=val_cfg.seed)
    occ_train.set_epoch(0)
    occ_val.set_epoch(0)
    rep = occ_train.structure_report()
    for line in rep["notes"]:
        logger.info(f"[occlusion] NOTE: {line}")

    train_dataset = TemporalWrapper.from_global_dataset(raw_train, train_cfg, occlusion=occ_train, with_rigid=True)
    val_dataset = TemporalWrapper(raw_val.deform_global, raw_train.template_xy, raw_train.joint_weights, config=val_cfg, occlusion=occ_val, rigid_world=raw_val.rigid_global, scale=train_dataset.scale)

    m_offset = train_cfg["max_radius"]
    w_size = 2*train_cfg.T_half+1
    model = build_temporal_model(cfg.model, num_keypoints=raw_train.N, window_size=w_size, max_offset=m_offset)
    loss = TemporalKeypointLoss(loss_cfg, train_dataset.scale)


    # Trainer
    return {
        "trainer_cls": Trainer,
        "trainer_kwargs": {
            "model": model,
            "loss_fn": loss,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "edges": raw_train.edges,
            "part_names": raw_train.part_names[1:],
            "behaviour_labels": raw_val.loco_clip,
            "config": training_cfg,
            "logger": logger,
            "device": device,
        },
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
    }