"""
@author xiaowei
UNFINISHED UNFINISHED UNFINISHED 
"""

import os
import random
import json
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import glob
import pandas as pd

class MaCVid(Dataset):
    """
    Data is structured as follows.
        |video_dataset_0
            |clip1.mp4
            |clip2.mp4
            |...
            |metadata.json
    """
    def __init__(self,
                 data_root,
                 resolution,
                 video_length,
                 frame_stride=4,
                 subset_split='all',
                 ):
        self.data_root = data_root
        self.resolution = resolution
        self.video_length = video_length
        self.subset_split = subset_split
        self.frame_stride = frame_stride
        assert(self.subset_split in ['train', 'test', 'all'])
        self.exts = ['avi', 'mp4', 'webm']

        if isinstance(self.resolution, int):
            self.resolution = [self.resolution, self.resolution]
        assert(isinstance(self.resolution, list) and len(self.resolution) == 2)

        self._make_dataset()
    
    def _make_dataset(self):
        if self.subset_split == 'all':
            data_folder = self.data_root
        else:
            data_folder = os.path.join(self.data_root, self.subset_split)
        
        
        # TODO param the threshold and filename
        metadata_path = os.path.join(data_folder,'metadata_catpion.json')
        with open(metadata_path, 'r') as f:
            self.videos = json.load(f)
        self.videos = [item for item in self.videos if item['basic']["clip_duration"] > 1.0 ]
        print(f'Number of videos = {len(self.videos)}')

    def __getitem__(self, index):
        while True:
            # video_path = os.path.join(self.data_root, f"videos/{self.videos.loc[index]['page_dir']}") + f"/{self.videos.loc[index]['videoid']}.mp4"
            # video_path = os.path.join(self.data_root, '.'+self.videos[index]['basic']['clip_path'])
            video_path = os.path.join(self.data_root, '.'+self.videos[index]['basic']['clip_path'])
            try:
                video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])
                if len(video_reader) < self.video_length:
                    index += 1
                    continue
                else:
                    break
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
    
        all_frames = list(range(0, len(video_reader), self.frame_stride))
        if len(all_frames) < self.video_length:
            all_frames = list(range(0, len(video_reader), 1))

        # select random clip
        rand_idx = random.randint(0, len(all_frames) - self.video_length)
        frame_indices = list(range(rand_idx, rand_idx+self.video_length))
        frames = video_reader.get_batch(frame_indices)
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'

        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        assert(frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        data = {'video': frames, 'caption':self.videos.loc[index]['name']}
        return data
    
    def __len__(self):
        return len(self.videos)