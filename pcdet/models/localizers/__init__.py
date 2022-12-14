from .localizer3d_template import Localizer3DTemplate
from .pv_localizer import PVLocalizer
from .pv_rcnn_plusplus import PVRCNNPlusPlus

__all__ = {
    'Localizer3DTemplate': Localizer3DTemplate,
    'PVLocalizer': PVLocalizer,
    'PVRCNNPlusPlus': PVRCNNPlusPlus
}

def build_localizer(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
    