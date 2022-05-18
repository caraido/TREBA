from .treba_model import TREBA_model
from .vq_triplet_treba_model import VQTripletTREBA_model

model_dict = {
    'treba_model' : TREBA_model,
    'vq_triplet_treba_model': VQTripletTREBA_model
}


def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        raise NotImplementedError
