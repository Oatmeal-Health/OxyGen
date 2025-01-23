"""
More strict typing, management, loading of config parameters.
Replacemnt for command-line args.
"""

from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Model-specific settings"""
    type: str = 'unetr-pp-cube-64'
    feature_size: int = 16
    learning_rate: float = 1.5e-4

@dataclass
class TrainingConfig:
    """Training-specific settings"""
    batch_size: int = 8
    device_count: int = 2
    max_epochs: int = 1000
    train_batch_limit: int = 100
    val_batch_limit: int = 20

@dataclass
class DataStoreConfig:
    """Where to get the data"""
    tensor_root: str = '' # Root directory of tensors with scans ??? needed ???
    train_scans: str = '' # Name of the file with training scans
    val_scans: str = '' # Name of the file with validation scans
    test_scans: str = '' # Name of the file with test scans
    output_dir: str = 'Output' # Name of output directory

@dataclass
class DataConfig:
    """How to use stored data in training process"""
    dataset_size: int = 50000
    subvolume_size: int = 64
    mask_ratio: float = 0.5
    mask_size: int = 10

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data_store: DataStoreConfig = field(default_factory=DataStoreConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Print out the config if needed for debugging.
# https://medium.com/@hitorunajp/configs-in-yaml-and-hydra-a71dc116be8a
if __name__ == "__main__":
    import sys
    import hydra
    from omegaconf import OmegaConf
    from hydra.core.config_store import ConfigStore

    # This is needed to fetch default values from the data classes
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)

    @hydra.main(config_path='./configs', config_name=None, version_base=None)
    def main(cfg: Config):
        print(OmegaConf.to_yaml(cfg))  # Print the configuration

    assert '--config-name' in ' '.join(sys.argv), '"--config-name" should point to the config, such as "unetr'
    main()
