import torch
import hydra
import logging
import wandb

import numpy as np

from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchinfo import summary
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from MTT.dataset import MotionPredictionDataset
from MTT.loss import BiomechanicalPoseLoss
from MTT.trainer import Trainer

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

    wandb.init(project=config['experiment']['project'])

    # Log the configuration
    logger.info("Configuration:")
    logger.info(f"{config['experiment']['project']} : {config['experiment']['name']}")
    logger.info(config['experiment']['description'])

    # Create dataloaders
    train_dataset = MotionPredictionDataset(
        file_path = config['train_dataset']['file_path'],
        pred_dir  = config['train_dataset']['pred_dir'],
        T_in      = config['train_dataset']['T_in'],
        T_out     = config['train_dataset']['T_out'],
    )
    val_dataset = MotionPredictionDataset(
        file_path = config['val_dataset']['file_path'],
        pred_dir  = config['val_dataset']['pred_dir'],
        T_in      = config['val_dataset']['T_in'],
        T_out     = config['val_dataset']['T_out'],
    )
    batch_size = config['train_dataset']['batch_size']

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"N joints: {train_dataset.N_active}")
    logger.info(f"Batch size: {config['train_dataset']['batch_size']}")

    if config['model']['type'] == 'MLPSpatialCorrector':
        from MTT.SpatialDenoiser.model import DeformationCorrector
        model_kwargs = config['model']
        model = DeformationCorrector(
            history_window = train_dataset.T_in,
            **model_kwargs
        )
    elif config['model']['type'] == 'SpatialTransformerCorrector':
        from MTT.SpatialTransformer.model import DeformationCorrector
        model_kwargs = config['model']
        model = DeformationCorrector(
            history_window = train_dataset.T_in,
            **model_kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")
    summary(model)

    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'], betas=(0.9, 0.999))

    warmup_scheduler = LinearLR(optimizer, start_factor=config['training']['warmup_start_factor'], total_iters=config['training']['warmup_epochs'])
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['total_epochs'] - config['training']['warmup_epochs'],eta_min=config['training']['min_learning_rate'])
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[config['training']['warmup_epochs']])
    edges = train_dataset.edge_index.T.tolist()
    loss = BiomechanicalPoseLoss(joint_edges     = edges,
                                 enable_skeleton = config['loss']['skeleton'],
                                 enable_temporal = config['loss']['temporal'],
                                 lambda_coord    = config['loss']['lambda_coord'],
                                 lambda_bone     = config['loss']['lambda_bone'],
                                 lambda_vel      = config['loss']['lambda_vel'])
    
    trainer = Trainer(model,
                      train_dataset,
                      val_dataset,
                      optimizer,
                      loss,
                      scheduler,
                      batch_size,
                      device,
                      output_dir  = config['training']['output_dir'],
                      log_wandb   = True,
                      patience    = config['training']['patience'],
                      grad_clip   = config['training']['grad_clip'],
                      joint_edges = edges)
    
    trainer.train(config['training']['total_epochs'])


if __name__ == "__main__":
    main()