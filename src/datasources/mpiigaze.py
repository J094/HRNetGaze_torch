import pathlib
import logging
from torch.utils import data

from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
import h5py


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MPIIGaze(Dataset):
    def __init__(self,
                 dataset_path: pathlib.Path,
                 cfg,
                 person_id: int = None) -> None:
        super(MPIIGaze).__init__()
        self.eye_image_shape = cfg['MODEL']['IMAGE_SIZE']
        self.dataset_path = dataset_path
        
        with h5py.File(self.dataset_path, 'r') as h5f:
            if person_id is None:
                self.len = h5f['image'].shape[0]
                self.person_id = None
            else:
                self.person_id = f'p{person_id:02}'
                self.len = h5f[self.person_id]['image'].shape[0] - 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        entry = dict()
        with h5py.File(self.dataset_path, 'r') as h5f:
            if self.person_id is None:
                eye = h5f['image'][idx]
                gaze = h5f['gaze'][idx]
            else:
                eye = h5f[self.person_id]['image'][idx]
                gaze = h5f[self.person_id]['gaze'][idx]
        eye = eye.astype(np.float32)
        interpolation = cv.INTER_CUBIC
        eye = cv.resize(eye, 
                        dsize=(self.eye_image_shape[1], self.eye_image_shape[0]), 
                        interpolation=interpolation)
        # eye *= 2.0 / 255.0
        # eye -= 1.0
        eye = np.expand_dims(eye, -1)
        # To NCWH.
        eye = np.transpose(eye, (2, 0, 1))
        entry['eye'] = eye / 255
        entry['gaze'] = gaze
        return entry
