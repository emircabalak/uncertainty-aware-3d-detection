"""Models for evidential 3D detector with pretrained PointPillars backbone."""

from .pretrained_backbone import (
    PillarVFE,
    PointPillarScatter,
    BaseBEVBackbone,
    PretrainedPointPillarsBackbone,
)
from .pretrained_loader import (
    download_pretrained,
    load_pretrained_backbone,
    freeze_backbone,
    unfreeze_all,
)
from .evidential_head import EvidentialCenterHead
from .uncertainty_nms import uncertainty_aware_nms
from .uncertainty_detector import EvidentialPretrainedDetector

__all__ = [
    "PillarVFE",
    "PointPillarScatter",
    "BaseBEVBackbone",
    "PretrainedPointPillarsBackbone",
    "download_pretrained",
    "load_pretrained_backbone",
    "freeze_backbone",
    "unfreeze_all",
    "EvidentialCenterHead",
    "uncertainty_aware_nms",
    "EvidentialPretrainedDetector",
]
