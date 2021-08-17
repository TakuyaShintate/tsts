from .types import RawDataset

__all__ = ["infer_dataset_type"]


def infer_dataset_type(X: RawDataset) -> str:
    """Return dataset type.

    Parameters
    ----------
    X : DatasetType
        Dataset

    Returns
    -------
    str
        Dataset type
    """
    num_dims = X[0].dim() + 1
    if num_dims == 2:
        return "mn"
    elif num_dims == 3:
        return "lmn"
    else:
        raise ValueError("Invalid dataset")
