import sys
import argparse

import torch.backends.cudnn as cudnn


def train(batch_size, num_workers, start_epoch, epochs, version):

    cudnn.benchmark = True
    cudnn.deterministic = False

    from src.configs import cfg
    import src.models.gaze_hrnet as gaze_hrnet
    import src.models.gaze_frame_net as gaze_frame_net
    model_hrnet = gaze_hrnet.get_gaze_net(cfg, pretrained="./models/model-v0.2-hrnet-epoch-50-loss-1.08801.pth")
    # Fix model_hrnet.
    for param in model_hrnet.parameters():
        param.requires_grad = False
    model_frame = gaze_frame_net.get_frame_net(pretrained="./models/model-v0.5-frame_net-epoch-13-loss-3.78204.pth", high_resolution=cfg['MODEL']['HIGH_RESOLUTION'])

    from torch.utils.data import DataLoader
    from src.datasources.unityeyes import UnityEyesDataset
    from src.datasources.mpiigaze import MPIIGaze
    # dataset_path = 'G:/Datasets/UnityEyes_Windows/480x640'
    # # dataset_path = '/home/junguo/Datasets/UnityEyes/480x640'
    # train_dataset = UnityEyesDataset(dataset_path, cfg, is_train=True, random_difficulty=True, generate_heatmaps=True)
    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    # val_dataset = UnityEyesDataset(dataset_path, cfg, is_train=False, random_difficulty=False, generate_heatmaps=False)
    # val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    train_dataset_path = 'G:/Datasets/MPIIGaze/Evaluation Subset/MPIIGazeCalibrationAll.h5'
    # train_dataset_path = './MPIIGazeCalibrationPer300.h5'
    train_dataset = MPIIGaze(dataset_path=train_dataset_path, cfg=cfg, person_id=0)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset_path = 'G:/Datasets/MPIIGaze/Evaluation Subset/MPIIGazeCalibrationAllEval.h5'
    # val_dataset_path = './MPIIGazeCalibrationPerEval.h5'
    val_dataset = MPIIGaze(dataset_path=val_dataset_path, cfg=cfg, person_id=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    from src.trainers.frame_net_trainer import FrameNetTrainer
    trainer = FrameNetTrainer(
        model_hrnet=model_hrnet,
        model_frame=model_frame,
        train_dataset=train_dataloader,
        val_dataset=val_dataloader,
        epochs=epochs,
        initial_learning_rate=0.0001,
        start_epoch=start_epoch,
        print_freq=64,
        version=version,
        tensorboard_dir='./logs'
    )

    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-b', type=int, required=True)
    parser.add_argument('--num-workers', '-w', type=int, required=True)
    parser.add_argument('--start-epoch', '-s', type=int, required=True)
    parser.add_argument('--end-epoch', '-e', type=int, required=True)
    args = parser.parse_args()

    batch_size = args.batch_size
    num_workers = args.num_workers
    start_epoch = args.start_epoch
    epochs = args.end_epoch

    train(batch_size, num_workers, start_epoch, epochs, version='v0.5-frame_net-calibration_mpii_All')