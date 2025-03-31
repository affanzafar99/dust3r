import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import os.path as osp
from PIL import Image
import cv2

import glob

import sys
sys.path.append("/home/user/zafara1/dust3r")
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import resize_img
from dust3r.datasets.base.base_stereo_view_dataset import view_name
from dust3r.viz import SceneViz, auto_cam_size
from dust3r.utils.image import rgb

from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True


class AISDataset(BaseStereoViewDataset):
    def __init__(self, *args, ROOT,method=None , **kwargs):
        self.ROOT = ROOT
        self.method = method 
        super().__init__(*args, **kwargs)
        self.scenes= []
        self.pairs =[]
        self.frames =[]
        self._load_data()

        
    def _load_data(self):
        self.scene_files = sorted(glob.glob(osp.join(self.ROOT,self.method, "dataset_s*.npz")))
        for scene_id, scene_file in enumerate(self.scene_files):
            with np.load(scene_file, allow_pickle=True) as data:
                frames = dict(data)
                self.frames.append(frames)
                self.scenes.append(scene_file) 
                for i in range(len(frames) - 1): 
                    if f"{i+1}" in frames and f"{i}" in frames : 
                        fm1_id = frames[f"{i}"].item()["img_number"]
                        fm2_id = frames[f"{i+1}"].item()["img_number"]
                        if fm2_id == fm1_id + 1:
                            self.pairs.append((scene_file,scene_id, fm1_id, fm2_id))                
    
    def get_view(self,pair_idx, resolution, rng):
        seq_path,seq, fm1, fm2 = self.pairs[pair_idx]
        seq_frames = self.frames[seq]
        views = []
        for view_index in [fm1, fm2]:
            data = seq_frames[f"{view_index}"].item()
            IR_img_path = data["IR_aligned_path"]
            ir_img = Image.open(str(IR_img_path))
            rgb_path = data["RGB_path"]
            rgb_image = Image.open(str(rgb_path))
            ir_img = resize_img(ir_img,size =224)
            depthmap = data["Depth"]
            intrinsics =np.float32( data["Camera_intrinsic"])
            camera_pose = np.float32(data["camera_pose"])     
            views.append(dict(
                img=ir_img,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='freiburg',
                label=str(seq_path),
                instance=str(IR_img_path)))
        return views 
        
    
    def _get_views(self, pair_idx, resolution, rng):
        
        views = self.get_view(pair_idx, resolution, rng)
        return views        

    def __len__(self):
        """Returns the number of samples in the dataset."""             
        return len(self.pairs)
        

if __name__ == "__main__":

    
    train_ds = AISDataset(ROOT="/home/user/zafara1/ThermalImages/test_data",method="Procrustes", split = "Train",resolution=224, aug_crop=16)
    views = train_ds[0]
    print(views)
