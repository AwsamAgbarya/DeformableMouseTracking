
import torch

from utils.dataset import TemporalDataset
from utils.masker import Masker

from OAT.dataset import TemporalWrapper
from OAT.models import GlobalAttentionModel, TemporalModel
from OAT.trainer import Trainer
from OAT.loss import ModularKeypointLoss

import hydra
from omegaconf import DictConfig, OmegaConf
@hydra.main(config_path="../configs", config_name="config.yaml", version_base="1.1")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Configuration:")
    print(f"{config['experiment']['project']} : {config['experiment']['name']}")
    print(config['experiment']['description'])
    print("=" * 80)

    train_config = config["train_dataset"]
    val_config = config["validation_dataset"]
    raw_train_dataset = TemporalDataset(file_path=train_config["file_path"], config=train_config)
    raw_val_dataset = TemporalDataset(file_path=val_config["file_path"], config=val_config)


    train_dataset = TemporalWrapper(
        data_d=raw_train_dataset.data_d,
        relative_dist=raw_train_dataset.relative_dist,
        outlier_flags=raw_train_dataset.outlier_mask,
        T_half=train_config.get("T_half", 8),
        max_radius=train_config.get("max_radius", 32),
        dropout_min=train_config.get("dropout_min", 0.10),
        dropout_max=train_config.get("dropout_max", 0.20),
        clean_frac=train_config.get("clean_frac", 0.15),
        pattern_weights=train_config.get("pattern_weights", None),
        seed=config["training"]["seed"],
    )
    val_dataset = TemporalWrapper(
        data_d=raw_val_dataset.data_d,
        relative_dist=raw_val_dataset.relative_dist,
        outlier_flags=raw_val_dataset.outlier_mask,
        T_half=val_config.get("T_half", 8),
        max_radius=val_config.get("max_radius", 32),
        dropout_min=val_config.get("dropout_min", 0.10),
        dropout_max=val_config.get("dropout_max", 0.20),
        clean_frac=train_config.get("clean_frac", 0.15),
        pattern_weights=val_config.get("pattern_weights", None),
        seed=config["training"]["seed"] + 1,
    )
    print(f"Processing train {len(train_dataset)} and validation {len(val_dataset)} windows")

    edges = raw_train_dataset.edges.tolist()
    lengths = raw_train_dataset.d2d_bone_lengths.tolist()
    bone_lengths =  {(p, c): l for (p, c), l in zip(edges, lengths)}

    num_keypoints = raw_train_dataset.N
    window_size = 2 * train_config.get("T_half", 8) + 1
    max_offset = train_config.get("max_radius", 8)
    masking_cfg = config["masking"]
    masker = Masker(
        dimensions=(config["training"]["batch_size"], window_size, num_keypoints),
        mask_strategy=masking_cfg["mask_strategy"],
        mask_min=masking_cfg["mask_min"],
        mask_max=masking_cfg["mask_max"],
        warmup_epochs=masking_cfg["warmup_epochs"],
        seed=config["training"]["seed"],
        # parent=raw_train_dataset.parent,
        bone_lengths=bone_lengths,
        n_control=masking_cfg.get("n_control", 3),
        corr_tau=masking_cfg.get("corr_tau", 1.5),
        random_frac=masking_cfg.get("mixing_ratio", 0.3),
    )

    # Model
    model_config = config["model"]
    model = GlobalAttentionModel(model_config, num_keypoints=num_keypoints, window_size=window_size)
    loss_fn = ModularKeypointLoss(config=config["loss"])

    # Trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        masker=masker,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        raw_train_dataset=raw_train_dataset,
        raw_val_dataset=raw_val_dataset,
        logger=None,
        config=config["training"],
        device=device,
        seed=config["training"]["seed"],
    )

    trainer.train()


if __name__ == "__main__":
    main()