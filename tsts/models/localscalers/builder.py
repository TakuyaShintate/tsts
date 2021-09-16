from torch.nn import Module
from tsts.cfg import CfgNode as CN
from tsts.core import LOCALSCALERS

__all__ = ["build_local_scaler"]


def build_local_scaler(
    num_in_feats: int,
    num_out_feats: int,
    cfg: CN,
) -> Module:
    """Build local scaler.

    Parameters
    ----------
    num_in_feats : int
        Number of input features

    num_out_feats : int
        Number of output features

    cfg : CN
        Global configuration

    Returns
    -------
    Module
        Forecasting model
    """
    local_scaler_name = cfg.LOCALSCALER.NAME
    cls = LOCALSCALERS[local_scaler_name]
    local_scaler = cls.from_cfg(
        num_in_feats,
        num_out_feats,
        cfg,
    )
    return local_scaler
