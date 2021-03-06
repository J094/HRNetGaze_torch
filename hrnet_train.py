import sys
import argparse
import torch.backends.cudnn as cudnn


def train(batch_size, num_workers, start_epoch, epochs, version):

    cudnn.benchmark = True
    cudnn.deterministic = False

    from src.configs import cfg
    import src.models.gaze_hrnet as gaze_hrnet
    model = gaze_hrnet.get_gaze_net(cfg, pretrained="")

    from torch.utils.data import DataLoader
    from src.datasources.unityeyes import UnityEyesDataset
    dataset_path = 'G:/Datasets/UnityEyes_Windows/480x640'
    # dataset_path = '/home/junguo/Datasets/UnityEyes/480x640'

    train_dataset = UnityEyesDataset(dataset_path, cfg, is_train=True, random_difficulty=True, generate_heatmaps=True)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    val_dataset = UnityEyesDataset(dataset_path, cfg, is_train=False, random_difficulty=False, generate_heatmaps=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    from src.trainers.hrnet_trainer import HRNetTrainer
    trainer = HRNetTrainer(
        model=model,
        train_dataset=train_dataloader,
        val_dataset=val_dataloader,
        epochs=epochs,
        initial_learning_rate=0.00001,
        start_epoch=start_epoch,
        print_freq=1,
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

    train(batch_size, num_workers, start_epoch, epochs, version='v0.3-hrnet')
