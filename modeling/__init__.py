# encoding: utf-8
from .baseline import Baseline


def build_model(arg, num_classes):
    model = Baseline(num_classes, arg.last_stride, arg.model_path, arg.model_name, arg.neck, arg.neck_feat, arg.kd)
    return model
