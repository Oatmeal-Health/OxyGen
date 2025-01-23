"""
Oatmeal Health adaptation of Swin UNETR for the SSL scenario.
"""

from monai.networks.nets.swin_unetr import SwinUNETR

from unetr_ssl import UNETR_BASE_SSL

class Swin_UNETR_SSL(UNETR_BASE_SSL):
    def __init__(self, learning_rate: float, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)

    def _get_unetr(self, feature_size):
        return SwinUNETR(
            img_size=64,  # unused but must be there
            in_channels=1,
            out_channels=1,
            feature_size=feature_size,
            downsample="merging",
            use_v2=True,            # original default -- False
            use_checkpoint=True,    # original default -- False
            spatial_dims=3,
            # TODO: explore other parameters later.
            # See GPT notes on the effect: https://chatgpt.com/share/6761c5d6-8c20-800b-8f45-9a1a4d77d71d
            # norm_name: tuple | str = "instance",
            # depths: Sequence[int] = (2, 2, 2, 2),
            # num_heads: Sequence[int] = (3, 6, 12, 24),
            # drop_rate: float = 0.0,
            # attn_drop_rate: float = 0.0,
            # dropout_path_rate: float = 0.0,
            # normalize: bool = True,
        )
