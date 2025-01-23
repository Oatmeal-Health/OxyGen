"""
Oatmeal Health modification of the original UNETR++ that works with volumes of 65x64x64.
Adapted for the SSL scenario.
"""

import extra_paths
from unetr_pp.network_architecture.cube_64x64x64.unetr_pp_cube_64x64x64 import UNETR_PP

from unetr_ssl import UNETR_BASE_SSL


class UNETR_PP_CUBE_64_SSL(UNETR_BASE_SSL):
    def __init__(self, learning_rate: float, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)

    def _get_unetr(self, feature_size):
        return UNETR_PP(
            in_channels=1,
            out_channels=1,
            feature_size=feature_size,  # default 16
            num_heads=16,
            depths=[10, 10, 10, 10],
            dims=[32, 64, 128, 256],
            do_ds=True,
            # TODO: explore later
            # hidden_size: int = 256,
            # pos_embed: str = "perceptron",
            # norm_name: Union[Tuple, str] = "instance",
            # dropout_rate: float = 0.0,
            # conv_op=nn.Conv3d,
        )
