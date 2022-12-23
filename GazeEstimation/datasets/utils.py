from .rgbddataset import load_dataset as load_rgbddataset


def get_dataset(data_dir, config, activity=None):
    if config['DATA']['NAME'] == 'RGBDGaze_dataset':
        return load_rgbddataset(data_dir, config, activity)
    else:
        raise ValueError(f'Unexpected dataset type: {config["DATA"]["NAME"]=}')
