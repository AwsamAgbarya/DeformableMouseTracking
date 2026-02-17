import torch
import hydra
import logging
from torchinfo import summary
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from utils.geometry import get_predefined_cams
from MVT.dataset import MV_Dataset, Pose_data
from MVT.models import MultiView3DKeypointModel
from MVT.trainer import Trainer
from MVT.loss import KeypointLoss, CurriculumScheduler
from MVT.metrics import MetricsEMA, MetricsTracker

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
    Ps = []
    for cam in dataset_config['cameras']:
        P = get_predefined_cams(cam)
        Ps.append(P)
    projection = torch.stack(Ps, dim=0)
    dataset = Pose_data(dataset_config)
    poses, com = dataset.get_unique_poses()

    # Generate indices and split for train/val
    dataset_size = len(poses)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor((1-dataset_config['train_ratio']) * dataset_size))
    train_indices, val_indices = indices[:split], indices[split:]
    train_dataset = MV_Dataset(poses[train_indices], com[train_indices], projection, n_rotations=dataset_config['n_rotations'], normalize=True, save_norm_path=dataset_config['norm_path'])
    val_dataset = MV_Dataset(poses[val_indices], com[val_indices], projection, n_rotations=dataset_config['n_rotations'], normalize=True, load_norm_path=dataset_config['norm_path'])

    # Prepare the model
    model_config = config['model']
    num_keypoints = dataset.part_count
    num_views = train_dataset.view_count

    model = MultiView3DKeypointModel(model_config, num_keypoints, num_views)
    summary(model)
    print(f"Processing train {len(train_dataset)} and validation {len(val_dataset)}")
    # Prepare the loss
    loss = KeypointLoss(projection, train_dataset.normalizer, loss_config=config['loss'],device=device)
    curriculum = CurriculumScheduler(config)
    train_metrics = MetricsTracker(num_keypoints, num_views)
    val_metrics = MetricsTracker(num_keypoints, num_views)
    metrics_ema = MetricsEMA(decay=config.get('ema_decay', 0.99))

    # Prepare the trainer
    trainer = Trainer(
        model=model,
        loss=loss,
        curriculum=curriculum,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        ema=metrics_ema,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        logger=logger,
        config=config['training'],
        device=device,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()