from tsts.cfg import CfgNode as CN
from tsts.core import TRANSFORMS
from tsts.transforms.pipeline import Pipeline


def build_pipeline(image_set: str, cfg: CN) -> Pipeline:
    if image_set == "train":
        transform_args = cfg.PIPELINE.TRANSFORMS_TRAIN
    else:
        transform_args = cfg.PIPELINE.TRANSFORMS_VALID
    transforms = []
    for arg in transform_args:
        transforms.append(TRANSFORMS[arg["name"]](**arg["args"]))
    pipeline = Pipeline(transforms)
    return pipeline
