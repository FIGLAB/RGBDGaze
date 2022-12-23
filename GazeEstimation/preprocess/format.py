import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

DATA_DIR = '/root/datadrive/RGBDGaze/dataset/RGBDGaze_dataset'
METADATA_PATH = f'{DATA_DIR}/metadata.pkl'
ACTIVITIES = ['standing', 'walking', 'sitting', 'lying']
INPUT_SIZE = (448, 448)


def get_all_participants():
    ret = []
    for p in Path(DATA_DIR).iterdir():
        if not p.is_dir():
            continue
        if not (p / 'decoded').is_dir():
            continue
        ret.append(str(p).split('/')[-1])
    return ret


def get_device_info(device, spec_df):
    info = {}
    # merge `iPhone 12` -> `iPhone12`
    if device.split('iPhone')[1][0] == ' ':
        device = 'iPhone' + device.split('iPhone')[1][1:]

    if device in spec_df.Name.tolist():
        for key in ['w_pt', 'w_cm', 'h_pt', 'h_cm']:
            info[key] = spec_df[spec_df['Name'] == device][key].tolist()[0]
    else:
        raise NotImplementedError(f'{device=} cannot be used')
    return info


def pt_to_cm(x_pt, y_pt, info):
    x_cm = x_pt / info['w_pt'] * info['w_cm']
    y_cm = y_pt / info['h_pt'] * info['h_cm']

    x_cm -= (info['w_cm']) / 2
    y_cm *= -1
    return x_cm, y_cm


def make_tensor(rgb, d, x, y, w, h):
    assert rgb.size > d.size
    if w != h:
        print(f'{w=} != {h=}')
        w = h = min(w, h)
    rgb = cv2.flip(rgb, 1)
    d = cv2.flip(d, 1)
    scale = rgb.shape[0] / d.shape[0]
    scaled_bbox_length = int(w/scale)
    scaled_rgb = cv2.resize(rgb, d.shape[::-1])
    concat = np.concatenate((scaled_rgb, d[:, :, np.newaxis]), axis=2)
    assert np.array_equal(concat[:, :, :3], scaled_rgb)
    concat = concat[int(x/scale):int(x/scale)+scaled_bbox_length,
                    int(y/scale):int(y/scale)+scaled_bbox_length, :]
    concat = cv2.resize(concat, INPUT_SIZE)
    # concat = cv2.flip(concat, 1)
    concat = concat.transpose((2, 0, 1))
    return torch.from_numpy(concat)


def process(pid, activity):
    print(f'==== {pid=} {activity=} ====')
    csv_file = f'{DATA_DIR}/{pid}/decoded/{activity}/label.csv'
    df = pd.read_csv(csv_file)
    spec_df = pd.read_csv(f'{DATA_DIR}/iphone_spec.csv')
    ret = {
        'pid': [],
        'device': [],
        'activity': [],
        'frameIndex': [],
        'labelDotX': [],
        'labelDotY': [],
        'imuX': [],
        'imuY': [],
        'imuZ': [],
    }
    pbar = tqdm(total=len(df))
    device_info = None
    if os.path.isdir(f'{DATA_DIR}/{pid}/tensor'):
        print('remove existing tensors')
        os.system('rm -rf {DATA_DIR}/{pid}/tensor')
    for (i, row) in df.iterrows():
        pbar.update(1)
        uid = row['uid']
        x, y, w, h = int(row['bbox_x']), int(row['bbox_y']), int(row['bbox_w']), int(row['bbox_h'])
        rgb_path = f'{DATA_DIR}/{pid}/decoded/{activity}/rgb/{uid}.jpg'
        depth_path = f'{DATA_DIR}/{pid}/decoded/{activity}/depth/{uid}.jpg'
        try:
            rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            tensor = make_tensor(rgb_img, depth_img, x, y, w, h)
        except Exception as e:
            print(e)
            continue
        save_tensor_path = f'{DATA_DIR}/{pid}/tensor/{activity}/{uid}.pt'
        if not os.path.isdir(os.path.dirname(save_tensor_path)):
            os.system(f'mkdir -p {os.path.dirname(save_tensor_path)}')
        torch.save(tensor, save_tensor_path)

        if device_info is None:
            device = row['device']
            device_info = get_device_info(device, spec_df)
        gt_x_cm, gt_y_cm = pt_to_cm(row['gt_x_pt'], row['gt_y_pt'], device_info)

        ret['pid'].append(pid)
        ret['device'].append(device)
        ret['activity'].append(activity)
        ret['frameIndex'].append(uid)
        ret['labelDotX'].append(gt_x_cm)
        ret['labelDotY'].append(gt_y_cm)
        ret['imuX'].append(float(row['imu_x']))
        ret['imuY'].append(float(row['imu_y']))
        ret['imuZ'].append(float(row['imu_z']))

    return ret


def update_metadata(org, new):
    for k in new.keys():
        if k in org.keys():
            org[k].extend(new[k])
        else:
            org[k] = new[k]
    return org


def main():
    participants = get_all_participants()
    print('participants: ', participants)

    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
    else:
        metadata = {}

    for pid in participants:
        for activity in ACTIVITIES:
            sub_metadata = process(pid, activity)
            update_metadata(metadata, sub_metadata)
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)


if __name__ == '__main__':
    main()
