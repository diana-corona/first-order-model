import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
import glob

#Adding syncnet
import torch
import cv2
import audio
from hparams import hparams
from os.path import dirname, join, basename, isfile
#

def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        directory = sorted(os.listdir(name))
        frames = []
        for file in directory:
          if not(file[-4:] == ".wav"):
            frames.append(file)

        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            #print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            #print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        #Adding syncnet values
        syncnet_T = 5
        syncnet_mel_step_size = 16
        #
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)

        def all_frames_exist(frames_list,forward_value,pth):
            for frame_id in range(frames_list[0], frames_list[1] + forward_value):
                frame = os.path.join(pth,str(frame_id)+'.jpg')
                if not os.path.exists(frame):
                    return False 
            return True 
         
        def get_new_frames_id(frames_list):
            fram_idx = np.random.choice(frames_list, replace=True, size=2)
            fram_idx = np.array([int(fram_idx[0]),int(fram_idx[1])])
            return fram_idx     

        def get_video_array(pth,frames_ids):     
            image_nam = []
            for fr in frames_ids:
                image_nam.append(os.path.join(pth,str(fr)+'.jpg'))   
            video_array = []
            for im in image_nam:
                if os.path.exists(im):
                    video_array.append(img_as_float32(io.imread(im)))
            return image_nam,video_array

        def get_window(start_frame_path,framx_idx):
            start_id = framx_idx
            vidname = start_frame_path    
            window_fnamesx = []
            for frame_id in range(start_id, start_id + syncnet_T):
                frame = os.path.join(path,str(frame_id)+'.jpg')
                if os.path.exists(frame):
                    if not isfile(frame):
                        print("get_window Image is none",frame )
                        return None
                    window_fnamesx.append(frame)
                else:
                    print(" Image dont exists",frame )
                    return None
            #returns an array of 5 consecutive frames (images paths) 
            #print(window_fnamesx)  
            return window_fnamesx

        def read_window(window_fnamesx):
            if window_fnamesx is None: return None
            windowx = []
            for fname in window_fnamesx:
                img = img_as_float32(io.imread(fname))
                if img is None:
                    print("read_window Image is none",fname )
                    return None

                windowx.append(img)
            #here we have 5 images of size (h, w, 3) so we have (5, h, w, 3)
            #print("read_window shape: ",len(windowx)," of ",windowx[0].shape)
            return windowx
        
        
        #---add mel 
        def crop_audio_window (spec, start_frame_num):
            start_idx = int(80. * (start_frame_num / float(hparams.fps)))
            end_idx = start_idx + syncnet_mel_step_size
            return spec[start_idx : end_idx, :]

        
        #---add indiv_mels 
        def get_segmented_mels(spec,start_frame_num):
            mels = []
            assert syncnet_T == 5
            start_frame_num = start_frame_num +1# 0-indexing ---> 1-indexing

            if start_frame_num - 2 < 0:
                print("start_frame_num - 2 < 0")
                return None
            for i in range(start_frame_num, start_frame_num + syncnet_T):
                m = crop_audio_window(spec, i - 2)
                #print("i : ", i, "m.shape[0] : ",m.shape[0])
                if m.shape[0] != syncnet_mel_step_size:
                    #print("m.shape[0] != syncnet_mel_step_size",start_frame_num,path)
                    return None
                mels.append(m.T)

            mels = np.asarray(mels)
            return mels

        if self.is_train and os.path.isdir(path):
            directory = os.listdir(path)
            frames = []
            for file in directory:
                if not(file[-4:] == ".wav"):
                    frames.append(file)
            
            frames_copy = np.sort([frame.replace('.jpg','') for frame in frames])
           
            #print("image_names",image_names,"path",path)
        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            video_array = video_array[frame_idx]

        
        out = {}
        
        #Adding syncnet values

        if self.is_train:
            print("path", path)
            if len(frames_copy) == 0:
              out['name'] = video_name
              out['indiv_mels'] = out['name']
              out['window_driving'] = out['name']
              out['mel'] = out['name']
              return out 
        
            frame_idx = get_new_frames_id(frames_copy)
            image_names, video_array = get_video_array(path,frame_idx)

            
            #driving 5 frames, source is one image only 
            window_driving = get_window([image_names[1]],frame_idx[1])
            out['window_driving'] = window_driving
            source = []
            window_driving_new = []
            if window_driving is not None:
                window_driving = read_window(window_driving)
                if self.transform is not None:
                    window_driving = self.transform(window_driving)
                
                window_driving = np.array(window_driving)
                window_driving =  window_driving.transpose((0, 3, 1, 2))
                out['window_driving'] = np.array(window_driving, dtype='float32')

            if len(source) == 0:
                for i in range(syncnet_T):
                    source.append(video_array[0])

            wavpath = join(path, "audio.wav")
            wav = audio.load_wav(wavpath, hparams.sample_rate)

            orig_mel = audio.melspectrogram(wav).T
            #mel  sould be from deriving window in frame id 1 , souce is frame id 0
            mel = crop_audio_window(orig_mel.copy(),frame_idx[1])
            out['mel'] = np.expand_dims(mel.T,0)
            indiv_mels = get_segmented_mels(orig_mel.copy(), frame_idx[1])
            out['indiv_mels'] = np.expand_dims(indiv_mels,1)

            #print("out['x'] ", len(out['x']) )
            if self.transform is not None:
                source = self.transform(source)

            source = np.array(source)
            source =  source.transpose((0, 3, 1, 2))
            out['window_source'] = np.array(source, dtype='float32')
            
            
        else:
            if self.transform is not None:
                video_array = self.transform(video_array)

            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        
        out['name'] = video_name

        if 'window_driving' in out:
            if indiv_mels is None:
                out['indiv_mels'] = out['name']
            if mel is None:
                out['mel'] = out['name']
            if window_driving is None:
                out['window_driving'] = out['name']

                
        #print("out ", out.keys(), len(out),type(out))
        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}