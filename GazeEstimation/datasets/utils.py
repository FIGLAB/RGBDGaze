from .rgbddataset import load_dataset as load_rgbddataset


def get_dataset(data_dir, config):
    if config['DATA']['NAME'] == 'RGBDGaze_dataset':
        return load_rgbddataset(data_dir, config)
    else:
        raise ValueError(f'Unexpected dataset type: {config["DATA"]["NAME"]=}')
