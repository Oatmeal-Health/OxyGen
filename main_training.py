import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# model and data classes
from ct_data_loader import CTScanDataModule

from swin_unetr_ssl import Swin_UNETR_SSL
from unetr_pp_cube_64_ssl import UNETR_PP_CUBE_64_SSL


class MAEVisualizationCallback(pl.Callback):
    """Callback to visualize original, masked, and reconstructed volumes"""
    def __init__(self, every_n_epochs: int):
        super().__init__()
        self.every_n_epochs = every_n_epochs
 
    def plot_slice(self, volume, axis=1, slice_idx=None):
        """Plot a 2D slice from the middle of the volume"""
        if len(volume.shape) == 4:
            volume = np.squeeze(volume, axis=0)
        if slice_idx is None:
            slice_idx = volume.shape[axis] // 2
        if axis == 0:
            return volume[slice_idx, :, :]
        elif axis == 1:
            return volume[:, slice_idx, :]
        else:
            return volume[:, :, slice_idx]

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs > 0:
            return
        
        # Get the validation dataloader
        val_dataloader = trainer.val_dataloaders
        
        # Get a batch of data
        batch = next(iter(val_dataloader))
        assert isinstance(batch, dict)
        x = batch['x']
        x_masked = batch['x_masked']

        x_masked = x_masked.to(pl_module.device)

        # Get predictions
        with torch.no_grad():
            pred = pl_module(x_masked)

        # Convert tensors to numpy arrays for visualization
        to_numpy = lambda y: y[0].cpu().numpy()
        original = to_numpy(x)
        reconstructed = to_numpy(pred)
        masked = to_numpy(x_masked)

        # Create figure
        fig, axes = plt.subplots(3, 3, figsize=(45, 45))    
        
        # Plot slices from different orientations
        to_plot = [(original[0], 'Original'), (masked[0], 'Masked'), (reconstructed[0], 'Reconstructed')]
        for i, axis in enumerate(['Sagittal', 'Coronal', 'Axial']):
            for j, (image, image_type) in enumerate(to_plot):
                axes[j, i].imshow(self.plot_slice(image, axis=i), cmap='gray')
                axes[j, i].set_title(f'{image_type} {axis}')

        # Log figure to tensorboard
        trainer.logger.experiment.add_figure(
            'reconstruction',
            fig,
            trainer.current_epoch
        )
        plt.close(fig)


def train_mae(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    output_dir: str,
    max_epochs: int,
    device_count: int,
    limit_train_batches: int,
    limit_val_batches: int,
):
    """
    Main training function
    
    Args:
        model: Model to be trained
        data: Data accessor
        output_dir: Directory to save checkpoints and logs
        max_epochs: Maximum number of training epochs
        device_count: Number of GPUs to use; set specific GPUs via "export CUDA_VISIBLE_DEVICES=2,3"
        limit_train_batches: max number of samples per training batch,
        limit_val_batches: max number of samples per validation batch,
    """

    project_slug = "MAE-90k_scans-50percent_masked"

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_dir, f'{project_slug}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, 'checkpoints'),
            filename='epoch_{epoch:02d}-test_mean_masked_error_{Test_Mean_Masked_Error:.5f}',
            monitor='Test_Mean_Masked_Error',
            mode='min',
            save_top_k=10,
            save_last=True,
        ),
        EarlyStopping(
            monitor='Test_Mean_Masked_Error',
            patience=50,
            mode='min',
        ),
        LearningRateMonitor(logging_interval='epoch'),
        MAEVisualizationCallback(every_n_epochs=1),
    ]

    # Set up logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, 'logs'),
        name='mae_training',
    )

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=device_count,
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        callbacks=callbacks,
        logger=logger,
        precision='16-mixed',  # Use mixed precision for faster training
        gradient_clip_val=1.0,  # Prevent exploding gradients
        accumulate_grad_batches=1,  # Increase if needed for larger effective batch size
    )

    # Pre-Training Validation
    print("Starting pre-training validation...")
    try:
        val_results = trainer.validate(model, data)
        print(f"Pre-training validation results: {val_results}")
    except Exception as e:
        print(f"Pre-training validation skipped due to error: {e}")

    # Train the model
    trainer.fit(model, data)

    # Final evaluation
    test_error = None
    try:
        val_results = trainer.validate(model, data, ckpt_path='best')
        test_error = val_results[0].get('Test_Mean_Masked_Error', None)

        if test_error is None:
            print("Test_Mean_Masked_Error not found in validation results; skipping checkpoint comparison.")
        else:
            print(f"Final validation Test_Mean_Masked_Error: {test_error:.4f}")

        # Save final validation results
        with open(os.path.join(output_dir, 'final_results.txt'), 'w') as f:
            f.write(f'Final Test_Mean_Masked_Error: {test_error:.4f}\n')

    except Exception as e:
        print(f"Final evaluation skipped due to error: {e}")

    return {
        'model': model,
        'output_dir': output_dir,
        'Test_Mean_Masked_Error': val_results[0]['Test_Mean_Masked_Error'] if test_error else None,
    }


if __name__ == "__main__":
    import hydra
    from omegaconf import OmegaConf
    from config_mgmt import Config
    import sys

    @hydra.main(config_path='./configs', config_name=None, version_base=None)
    def main(cfg: Config):
        """
        Parses values and creates a full Config object.
        When "config_name" is "base_config", uses default values in Python code.
        When it points to the name of the file, such as "unetr" in "config_path", it uses those values.
        Both "config_path" and "config_name" can be set on the command line.
        """
        print(40*'*')
        print("USING THE FOLLOWING CONFIG")
        print()
        print(OmegaConf.to_yaml(cfg))  # Print the configuration
        print(40*'*')

        model_class_dict = {
            'unetr-pp-cube-64': UNETR_PP_CUBE_64_SSL,
            'swin-unetr': Swin_UNETR_SSL,
        }
        
        assert cfg.model.type in model_class_dict, f'Model type should be one of: {list(model_class_dict.keys())}'
        ModelClass = model_class_dict[cfg.model.type]
        model = ModelClass(
            feature_size=cfg.model.feature_size,
            learning_rate=cfg.model.learning_rate,
        )

        data = CTScanDataModule(cfg)

        train_mae(
            model=model,
            data=data,
            output_dir=cfg.data_store.output_dir,
            max_epochs=cfg.training.max_epochs,
            device_count=cfg.training.device_count,
            limit_train_batches=cfg.training.train_batch_limit,
            limit_val_batches=cfg.training.val_batch_limit,
        )


    assert '--config-name' in ' '.join(sys.argv), '"--config-name" should point to the YAML config file, such as "unetr.yaml"'
    main()
