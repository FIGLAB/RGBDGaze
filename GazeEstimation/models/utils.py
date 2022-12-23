from .spatial_weights_cnn import SpatialWeightsCNN
from .two_stream import TwoStream


def get_model(config):
    if config['MODEL']['TYPE'] == 'SpatialWeightsCNN':
        model = SpatialWeightsCNN(feature_type=config['MODEL']['FEATURE_TYPE'])
    elif config['MODEL']['TYPE'] == 'TwoStream':
        model = TwoStream()
    else:
        raise ValueError(f'Unexpected model type: {config["MODEL"]["TYPE"]=}')

    return model
