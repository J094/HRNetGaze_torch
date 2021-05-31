import argparse
import pathlib
import logging

import torch
import numpy as np
from torch.utils.data import DataLoader
import tqdm

import src.models.gaze_frame_net as gaze_frame_net
import src.utils.gaze as gaze_util


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', '-d', type=str, required=True)
    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset_path)

    model_hrnet = torch.load('./models/model-v0.2-hrnet-epoch-50-loss-1.08801.pth')
    model_hrnet.eval()
    model_frame = torch.load('./models/model-v0.2-frame_net-epoch-15-loss-3.07270.pth')
    model_frame.eval()

    from src.configs import cfg
    from src.datasources.mpiigaze import MPIIGaze

    angular_error_all = 0.0
    for person_id in tqdm.tqdm(range(15)):
        dataset = MPIIGaze(dataset_path=dataset_path, cfg=cfg, person_id=person_id)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

        angular_error = 0.0
        num_batches = 0.0
        for one_batch in tqdm.tqdm(dataloader):
            eye_input = one_batch['eye'].cuda()
            gaze_label = one_batch['gaze'].cuda()
            # Do prediction on gaze_hrnet.
            heatmaps_predict, ldmks_predict, radius_predict = model_hrnet(eye_input)
            # Get frames of eye image.
            frames_predict = gaze_frame_net.get_gaze_frame(heatmaps_predict, ldmks_predict)
            # Predict gaze by frame_net.
            gaze_predict = model_frame(frames_predict, radius_predict)
            loss = gaze_util.angular_error_torch(gaze_predict, gaze_label)
            error = loss.cpu().item()
            angular_error += error
            num_batches += 1

        mean_angular_error = angular_error / num_batches
        logger.info(f"\nMean angular error for Person{person_id} is {mean_angular_error}\n")

        angular_error_all += mean_angular_error

    mean_angular_error_all = angular_error_all / 15
    logger.info(f"\nMean angular error for MPIIGaze is {mean_angular_error_all}\n")


if __name__ == "__main__":
    main()