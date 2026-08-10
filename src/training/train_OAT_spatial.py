import torch
import hydra
import logging
from omegaconf import DictConfig, OmegaConf

from utils.dataset import SnapshotDataset
from OAT.dataset import SnapshotWrapper
from OAT.models import SnapshotModel
from OAT.trainer import Trainer
from OAT.loss import KeypointLoss, CurriculumScheduler
from OAT.metrics import MetricsEMA, MetricsTracker
from utils.masker import Masker

@hydra.main(config_path="configs", config_name="config.yaml", version_base="1.1")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Prepare the logger
    logger = logging.getLogger('Trainer')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('train.log', mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Log the configuration
    print("=" * 80)
    print("Configuration:")
    print(f"{config['experiment']['project']} : {config['experiment']['name']}")
    print(config['experiment']['description'])
    print("=" * 80)
    
    # Preparing the dataset
    dataset_config = config['train_dataset']
    raw_dataset = SnapshotDataset(dataset_config['file_path'], n_rotations = dataset_config['n_rotations'], reference_parts = dataset_config['reference_parts'],
        config={
            "validate":True,
            "normalize":True,
            "save_norm_path": dataset_config["norm_path"],
            "train_ratio": dataset_config["train_ratio"],
        }
    )
    train_dataset = SnapshotWrapper(raw_dataset.data[raw_dataset.train_indices])
    val_dataset = SnapshotWrapper(raw_dataset.data[raw_dataset.val_indices])

    # Prepare the model
    model_config = config['model']
    num_keypoints = raw_dataset.N
    masker = Masker(
        dimensions=(config['training']['batch_size'], num_keypoints),
        mask_strategy=config['curriculum']['masking_schedule']['scheme'],
        mask_min=config['curriculum']['masking_schedule']['start_ratio'],
        mask_max=config['curriculum']['masking_schedule']['end_ratio'],
        warmup_epochs=config['curriculum']['masking_schedule']['total_epochs'],
        seed=config['training']['seed'])

    model = SnapshotModel(model_config, num_keypoints=num_keypoints)
    print(f"Processing train {len(train_dataset)} and validation {len(val_dataset)}")
    # Prepare the loss
    loss = KeypointLoss(raw_dataset.normalizer, loss_config=config['loss'],device=device)
    curriculum = CurriculumScheduler(config)
    train_metrics = MetricsTracker(num_keypoints)
    val_metrics = MetricsTracker(num_keypoints)
    metrics_ema = MetricsEMA(decay=config.get('ema_decay', 0.99))

    # Prepare the trainer
    trainer = Trainer(
        model=model,
        loss=loss,
        curriculum=curriculum,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        ema=metrics_ema,
        raw_dataset=raw_dataset,
        masker=masker,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        logger=logger,
        config=config['training'],
        device=device,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()