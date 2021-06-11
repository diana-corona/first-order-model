import matplotlib

matplotlib.use('Agg')

import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset

from modules.generator import OcclusionAwareGenerator
from modules.discriminator import MultiScaleDiscriminator
from modules.keypoint_detector import KPDetector

import torch

from train import train
from reconstruction import reconstruction
from animate import animate

#add SyncNet
from modules.syncnet import SyncNet_color as SyncNet

if __name__ == "__main__":
    
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    #add SyncNet
    parser.add_argument('--syncnet_checkpoint_path', help='Load the pre-trained Expert discriminator', required=True, type=str)
    #
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

    if torch.cuda.is_available():
        generator.to(opt.device_ids[0])
    if opt.verbose:
        print(generator)

    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])
    if torch.cuda.is_available():
        discriminator.to(opt.device_ids[0])
    if opt.verbose:
        print(discriminator)
    

    #add SyncNet
    def load_checkpoint_path(checkpoint_path):
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        return checkpoint

    def load_checkpoint(path, model, optimizer, reset_optimizer=False):
        print("Load checkpoint from: {}".format(path))
        checkpoint = load_checkpoint_path(path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)
        if not reset_optimizer:
            optimizer_state = checkpoint["optimizer"]
            if optimizer_state is not None:
                print("Load optimizer state from {}".format(path))
                optimizer.load_state_dict(checkpoint["optimizer"])
        return model

    syncnet = SyncNet()
    
    for p in syncnet.parameters():
        p.requires_grad = False

    syncnet = load_checkpoint(opt.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True)

    
    if torch.cuda.is_available():
        syncnet.to(opt.device_ids[0])
    if opt.verbose:
        print(syncnet)
    #

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])


    
    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])

    if opt.verbose:
        print(kp_detector)

    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'train':
        print("Training...")
        train(config, generator, discriminator, kp_detector, opt.checkpoint, log_dir, dataset, opt.device_ids, syncnet)
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        reconstruction(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'animate':
        print("Animate...")
        animate(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)
