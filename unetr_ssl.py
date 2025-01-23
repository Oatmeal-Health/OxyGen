"""
Base class for UNETR-based models (such as Swin-UNETR, UNETR++)
"""

from abc import ABC, abstractmethod
import torch
import pytorch_lightning as pl

class UNETR_BASE_SSL(pl.LightningModule, ABC):
    def __init__(self, learning_rate: float, **kwargs):
        super().__init__()
        self.save_hyperparameters()    
        self.learning_rate = learning_rate
        self.unetr = self._get_unetr(**kwargs)

    @abstractmethod
    def _get_unetr(self, **kwargs):
        pass

    def forward(self, x):
        """
        Forward pass without masking (for inference)
        """
        return self.unetr(x)

    def training_step(self, batch: dict, batch_idx):
        x = batch['x']
        x_masked = batch['x_masked']
        mask = batch['mask']
       
        # Get reconstruction
        reconstruction = self.unetr(x_masked)
        if isinstance(reconstruction, list):
            reconstruction = reconstruction[0]

        n_masked_voxels = torch.sum(mask > 0)
        n_unmasked_voxels = torch.sum(mask == 0)
        
        masked_true   = mask > 0
        unmasked_true = mask == 0

        unmasked_reconstruction = reconstruction * unmasked_true 
        unmasked_ground_truth   = x * unmasked_true

        masked_reconstruction   = reconstruction * masked_true
        masked_ground_truth     = x * masked_true

        signed_unmasked_error   = unmasked_reconstruction - unmasked_ground_truth
        signed_masked_error     = masked_reconstruction - masked_ground_truth
        
        mean_unmasked_error = torch.sum(torch.abs(signed_unmasked_error)) / n_unmasked_voxels
        mean_masked_error   = torch.sum(torch.abs(signed_masked_error)) / n_masked_voxels

        mean_squared_unmasked_error = torch.sum(signed_unmasked_error ** 2) / n_unmasked_voxels
        mean_squared_masked_error   = torch.sum(signed_masked_error ** 2) / n_masked_voxels

        # accelerate focus on masked regions with Lagrandian coefficient
        # TODO: place this into a config
        unmasked_weight = 0.1
        masked_weight = 0.9
        # TODO: place this into a config
        total_loss = unmasked_weight * mean_squared_unmasked_error + masked_weight * mean_squared_masked_error
        self.log('Train_Mean_Masked_Error', mean_masked_error, prog_bar=True, sync_dist=True) 
        self.log('Train_Mean_Unmasked_Error', mean_unmasked_error, prog_bar=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch: dict, batch_idx):
        x = batch['x']
        x_masked = batch['x_masked']
        mask = batch['mask']
       
        # Get reconstruction
        reconstruction = self.unetr(x_masked)
        if isinstance(reconstruction, list):
            reconstruction = reconstruction[0]
 
        n_masked_voxels = torch.sum(mask > 0)
        n_unmasked_voxels = torch.sum(mask == 0)
        
        masked_true   = mask > 0
        unmasked_true = mask == 0
 
        unmasked_reconstruction = reconstruction * unmasked_true 
        unmasked_ground_truth   = x * unmasked_true

        masked_reconstruction   = reconstruction * masked_true
        masked_ground_truth     = x * masked_true

        signed_unmasked_error   = unmasked_reconstruction - unmasked_ground_truth
        signed_masked_error     = masked_reconstruction - masked_ground_truth
        
        mean_unmasked_error = torch.sum(torch.abs(signed_unmasked_error)) / n_unmasked_voxels
        mean_masked_error   = torch.sum(torch.abs(signed_masked_error)) / n_masked_voxels

        mean_squared_unmasked_error = torch.sum(signed_unmasked_error ** 2) / n_unmasked_voxels
        mean_squared_masked_error   = torch.sum(signed_masked_error ** 2) / n_masked_voxels
 
        # accelerate focus on masked regions with Lagrandian coefficient
        # TODO: place this into a config
        unmasked_weight = 0.1
        masked_weight = 0.9
        # TODO: place this into a config
        total_loss = unmasked_weight * mean_squared_unmasked_error + masked_weight * mean_squared_masked_error
        
        # Log validation metrics
        self.log('Test_Mean_Masked_Error', mean_masked_error, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('Test_Mean_Unmasked_Error', mean_unmasked_error, prog_bar=True, sync_dist=True)
        self.log('Test_Error', total_loss, prog_bar=True, sync_dist=True)

        return {
            'Test_Mean_Masked_Error': mean_masked_error,
            'Test_Mean_Unmasked_Error': mean_unmasked_error,
            'Test_Error': total_loss,
            'val_loss': total_loss,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.05
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "Test_Mean_Masked_Error",
                "interval": "epoch",
                "frequency": 1
            }
        }
