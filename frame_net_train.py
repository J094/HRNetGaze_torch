import sys
import torch.backends.cudnn as cudnn


def train(batch_size, num_workers, start_epoch, epochs, version):

    # cudnn.benchmark = True
    # cudnn.deterministic = True

    from src.configs import cfg
    import src.models.gaze_hrnet as gaze_hrnet
    import src.models.gaze_frame_net as gaze_frame_net
    model_hrnet = gaze_hrnet.get_gaze_net(cfg, pretrained="./models/model-v0.2-hrnet-radius-epoch-29-loss-0.43388.pth")
    # Fix model_hrnet.
    for param in model_hrnet.parameters():
        param.requires_grad = False
    model_frame = gaze_frame_net.get_frame_net(pretrained="")

    from torch.utils.data import DataLoader
    from src.datasources.unityeyes import UnityEyesDataset
    dataset_path = 'G:/Datasets/UnityEyes_Windows/480x640'
    # dataset_path = '/home/junguo/Datasets/UnityEyes/480x640'

    train_dataset = UnityEyesDataset(dataset_path, cfg, is_train=True, random_difficulty=True, generate_heatmaps=True)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    val_dataset = UnityEyesDataset(dataset_path, cfg, is_train=False, random_difficulty=False, generate_heatmaps=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    from src.trainers.frame_net_trainer import FrameNetTrainer
    trainer = FrameNetTrainer(
        model_hrnet=model_hrnet,
        model_frame=model_frame,
        train_dataset=train_dataloader,
        val_dataset=val_dataloader,
        epochs=epochs,
        initial_learning_rate=0.00001,
        start_epoch=start_epoch,
        print_freq=8,
        version=version,
        tensorboard_dir='./logs'
    )

    trainer.run()


if __name__ == "__main__":
    batch_size = eval(sys.argv[1])
    num_workers = eval(sys.argv[2])
    start_epoch = eval(sys.argv[3])
    epochs = eval(sys.argv[4])

    train(batch_size, num_workers, start_epoch, epochs, version='v0.2-frame_net')