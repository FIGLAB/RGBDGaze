import argparse
import logging
import os
import pickle
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from datasets import get_dataset
from models import get_model
from utils import AverageMeter, load_config

random.seed(2022)


logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)

stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)


def get_all_participants():
    ret = []
    for p in Path('/root/datadrive/RGBDGaze/dataset/RGBDGaze_dataset').iterdir():
        if not p.is_dir():
            continue
        if not (p / 'tensor').is_dir():
            continue
        ret.append(str(p).split('/')[-1])
    return ret


PARTICIPANTS = get_all_participants()


def get_args():
    parser = argparse.ArgumentParser(description='RGBD Gaze Estimation (pytorch).')
    parser.add_argument(
        '--project_dir',
        default='/root/datadrive/RGBDGaze',
        help='Path to project directory.',
    )
    parser.add_argument(
        '--config',
        required=True,
        default='./config/rgbd.yml',
        help='Path to config yaml.',
    )
    parser.add_argument(
        '--checkpoint',
        default='/root/datadrive/RGBDGaze/models/SpatialWeightsCNN_gazecapture/pretrained_rgb.pth',
        help='Path to checkpoint.',
    )
    args = parser.parse_args()
    return args


def get_dataloaders(data_dir, config):
    train, val, test = get_dataset(data_dir, config)

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=config['TRAIN']['DATALOADER']['BATCH_SIZE'],
        shuffle=config['TRAIN']['DATALOADER']['SHUFFLE'],
        num_workers=config['TRAIN']['DATALOADER']['NUM_WORKERS'],
        pin_memory=config['TRAIN']['DATALOADER']['PIN_MEMORY'],
    )

    val_loader = torch.utils.data.DataLoader(
        val,
        batch_size=config['TRAIN']['VALIDATION']['DATALOADER']['BATCH_SIZE'],
        shuffle=config['TRAIN']['VALIDATION']['DATALOADER']['SHUFFLE'],
        num_workers=config['TRAIN']['VALIDATION']['DATALOADER']['NUM_WORKERS'],
        pin_memory=config['TRAIN']['VALIDATION']['DATALOADER']['PIN_MEMORY'],
    )

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=config['TEST']['DATALOADER']['BATCH_SIZE'],
        shuffle=config['TEST']['DATALOADER']['SHUFFLE'],
        num_workers=config['TEST']['DATALOADER']['NUM_WORKERS'],
        pin_memory=config['TEST']['DATALOADER']['PIN_MEMORY'],
    )
    return train_loader, val_loader, test_loader


def get_criterion():
    return nn.MSELoss().cuda()


def get_optimizer(model, config):
    if config['TRAIN']['OPTIMIZER']['NAME'] == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['TRAIN']['OPTIMIZER']['PARAM']['LR'],
            momentum=config['TRAIN']['OPTIMIZER']['PARAM']['MOMENTUM'],
            weight_decay=config['TRAIN']['OPTIMIZER']['PARAM']['WEIGHT_DECAY'],
        )
    else:
        raise ValueError(f'Unexpected optimizer: {config["TRAIN"]["OPTIMIZER"]}')
    return optimizer


def get_device(config):
    assert config['TRAIN']['NUM_GPU'] == torch.cuda.device_count(), \
            f'Expected number of GPUs is {config["TRAIN"]["NUM_GPU"]}, ' \
            + f'but {torch.cuda.device_count()} are going to be used.'
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_scheduler(optimizer, config):
    return torch.optim.lr_scheduler.StepLR(optimizer, config['TRAIN']['OPTIMIZER']['PARAM']['STEP_SIZE'], gamma=0.1)


def load_checkpoint(fp):
    data = torch.load(fp)
    return data


def save_checkpoint(data, fp):
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    torch.save(data, fp)


def main():
    args = get_args()
    config = load_config(args.config)
    print(config)
    data_dir = os.path.join(args.project_dir, 'dataset', config['DATA']['NAME'])
    model_dir = os.path.join(args.project_dir, 'models', 'LOPO-'+config['MODEL']['NAME'])
    logger.info(f'Using data from {data_dir} and output result to {model_dir}')

    torch.backends.cudnn.benchmark = True  # input image size is fixed
    criterion = get_criterion()
    device = get_device(config)
    participants = PARTICIPANTS
    logger.info(f'Using {len(participants)} participants. LEAVE-ONE-PARTICIPANT-OUT')

    for pid in participants:
        logger.info('='*15)
        save_dir = os.path.join(model_dir, pid)
        if os.path.exists(save_dir):
            logger.info(f'already exist: skipping {pid}')
            continue
        else:
            os.makedirs(save_dir, exist_ok=False)

        handler = logging.FileHandler(filename=f'{save_dir}/log.log')
        logger.addHandler(handler)

        logger.info(f'Set {pid} as TEST and output result to {save_dir}')
        config['DATA']['TEST_PID'] = [pid]
        config['DATA']['VAL_PID'] = random.sample([p for p in participants if p != pid], 5)
        config['DATA']['TRAIN_PID'] = [p for p in participants if p != pid and p != config['DATA']['VAL_PID']]
        model = get_model(config)
        train_loader, val_loader, test_loader = get_dataloaders(data_dir, config)
        model = model.to(device)
        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)

        if args.checkpoint is not None:
            saved = load_checkpoint(args.checkpoint)
            model.load_pretrained_data(saved['state_dict'])
            best_prec1 = saved['best_prec1']
            del saved
            model.to(device)
            logger.info(f'Loaded checkpoint from {args.checkpoint}. \\ {best_prec1=}')

        max_epoch = config['TRAIN']['NUM_EPOCH']
        best_model = model
        best_prec1 = 1e20
        best_epoch = 0

        for ep in range(1, max_epoch+1):
            train(train_loader, model, criterion, optimizer, ep, device)
            scheduler.step()
            if ep % config['TRAIN']['VALIDATION']['VAL_INTERVAL'] == 0:
                prec1, _, _ = evaluate(val_loader, 'val', model, criterion, ep, device)
                if prec1 < best_prec1:
                    best_epoch = ep
                    best_prec1 = prec1
                    best_model = model
                if ep == 1:
                    os.system(f'cp {args.config} {model_dir}')

                test_prec1, _, _ = evaluate(test_loader, 'test', model, criterion, ep, device)
                with open(os.path.join(save_dir, f'val_loss_{ep}.txt'), 'w') as f:
                    f.write(str(prec1))
                with open(os.path.join(save_dir, f'test_loss_{ep}.txt'), 'w') as f:
                    f.write(str(test_prec1))

        save_checkpoint(
            {
                'epoch': best_epoch,
                'state_dict': best_model.state_dict(),
                'best_val_prec1': best_prec1,
            },
            os.path.join(save_dir, f'best.pth'),
        )

        loss, gt_list, pred_list = evaluate(test_loader, 'test', best_model, criterion, best_epoch, device)
        with open(os.path.join(save_dir, 'gt_list.pkl'), 'wb') as f:
            pickle.dump(gt_list, f)
        with open(os.path.join(save_dir, 'pred_list.pkl'), 'wb') as f:
            pickle.dump(pred_list, f)
        with open(os.path.join(save_dir, 'test_loss.txt'), 'w') as f:
            f.write(str(loss))
        logger.removeHandler(handler)


def train(loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, data in enumerate(loader):

        assert isinstance(data, list)
        assert len(data) == 2
        face, gaze = data
        face = face.to(device)
        output = model(face)

        n_batch = face.size(0)
        gaze = gaze.to(device)
        loss = criterion(output, gaze)
        losses.update(loss.data.item(), n_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        logger.info(
            'Epoch (train): [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(loader), batch_time=batch_time, loss=losses,
            ),
        )


@torch.no_grad()
def evaluate(loader, val_type, model, criterion, epoch, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    l2_errors = AverageMeter()

    # switch to evaluate mode
    model.eval()

    gt_list, pred_list = [], []
    end = time.time()
    for i, data in enumerate(loader):

        assert isinstance(data, list)
        assert len(data) == 2
        face, gaze = data
        face = face.to(device)
        output = model(face)

        gt_list.append(gaze)
        pred_list.append(output.to(torch.device('cpu')))

        n_batch = face.size(0)
        gaze = gaze.to(device)
        loss = criterion(output, gaze)
        l2_error = output - gaze
        l2_error = torch.mul(l2_error, l2_error)
        l2_error = torch.sum(l2_error, 1)
        l2_error = torch.mean(torch.sqrt(l2_error))
        losses.update(loss.data.item(), n_batch)
        l2_errors.update(l2_error.item(), n_batch)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        logger.info(
            'Epoch ({0}): [{1}][{2}/{3}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Error L2 {l2_error.val:.4f} ({l2_error.avg:.4f})\t'.format(
                val_type, epoch, i, len(loader), batch_time=batch_time, loss=losses, l2_error=l2_errors,
            ),
        )

    return l2_errors.avg, gt_list, pred_list


if __name__ == '__main__':
    main()
    print('DONE')
