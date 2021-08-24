from torch.nn import Module
from tsts.cfg import CfgNode as CN
from tsts.core import MODELS

__all__ = ["build_model"]


def build_model(
    num_in_feats: int,
    num_out_feats: int,
    cfg: CN,
) -> Module:
    """Build model.

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
    model_name = cfg.MODEL.NAME
    cls = MODELS[model_name]
    model = cls.from_cfg(
        num_in_feats,
        num_out_feats,
        cfg,
    )
    return model
