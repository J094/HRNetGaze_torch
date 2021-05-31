import os
import math
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import src.models.gaze_frame_net as gaze_frame_net

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameNetTrainer(object):
    def __init__(self,
                 model_hrnet,
                 model_frame,
                 train_dataset,
                 val_dataset,
                 epochs=100,
                 initial_learning_rate=0.001,
                 start_epoch=1,
                 print_freq=500,
                 version='v0.1',
                 tensorboard_dir='./logs'):
        super(FrameNetTrainer, self).__init__()
        self.version = version
        self.model_hrnet = model_hrnet
        self.model_frame = model_frame
        self.print_freq = print_freq

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.epochs = epochs
        self.current_learning_rate = initial_learning_rate
        self.start_epoch = start_epoch

        self.loss_obj = nn.MSELoss(reduction='mean')
        self.optimizer = self.optimizer = optim.Adam(self.model_frame.parameters(), lr=self.current_learning_rate, weight_decay=1e-4)
        if not os.path.exists(os.path.join(tensorboard_dir)):
            os.makedirs(os.path.join(tensorboard_dir))
        self.summary_writer_train = SummaryWriter(tensorboard_dir + f'/train-{self.version}')
        self.summary_writer_val = SummaryWriter(tensorboard_dir + f'/val-{self.version}')

        self.patience_count = 0
        self.max_patience = 2
        self.last_val_loss = math.inf
        self.lowest_val_loss = math.inf
        self.best_model = None

    def lr_decay(self):
        """
        This effectively simulate ReduceOnPlateau learning rate schedule. Learning rate
        will be reduced by a factor of 10 if there's no improvement over [max_patience] epochs
        """
        if self.patience_count >= self.max_patience:
            self.current_learning_rate /= 10.0
            self.patience_count = 0
        elif self.last_val_loss == self.lowest_val_loss:
            self.patience_count = 0
        self.patience_count += 1
        self.optimizer = optim.Adam(self.model_frame.parameters(), lr=self.current_learning_rate, weight_decay=1e-4)

    def compute_coord_loss(self, predict, label):
        loss = self.loss_obj(predict, label)
        return loss

    def compute_angular_loss(self, predict, label):
        """Pytorch method to calculate angular loss (via cosine similarity)"""
        def angle_to_unit_vectors(y):
            sin = torch.sin(y)
            cos = torch.cos(y)
            return torch.stack([
                cos[:, 0] * sin[:, 1],
                sin[:, 0],
                cos[:, 0] * cos[:, 1],
                ], dim=1)

        a = angle_to_unit_vectors(predict)
        b = angle_to_unit_vectors(label)
        ab = torch.sum(a*b, dim=1)
        a_norm = torch.sqrt(torch.sum(torch.square(a), dim=1))
        b_norm = torch.sqrt(torch.sum(torch.square(b), dim=1))
        cos_sim = ab / (a_norm * b_norm)
        cos_sim = torch.clip(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
        ang = torch.acos(cos_sim) * 180. / math.pi
        return torch.mean(ang)

    def train_step(self, inputs):
        eye_input = inputs['eye'].cuda()
        gaze_label = inputs['gaze'].cuda()

        heatmaps, ldmks, radius = self.model_hrnet(eye_input)
        frame = gaze_frame_net.get_gaze_frame(heatmaps, ldmks)

        with torch.set_grad_enabled(True):
            gaze_predict = self.model_frame(frame, radius)
            loss = self.compute_angular_loss(gaze_predict, gaze_label)
            self.model_frame.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def val_step(self, inputs):
        eye_input = inputs['eye'].cuda()
        gaze_label = inputs['gaze'].cuda()

        heatmaps, ldmks, radius = self.model_hrnet(eye_input)
        frame = gaze_frame_net.get_gaze_frame(heatmaps, ldmks)

        gaze_predict = self.model_frame(frame, radius)
        loss = self.compute_angular_loss(gaze_predict, gaze_label)

        return loss.item()

    def run(self):
        def train_epoch(dataset):
            msg = 'Start training...'
            logger.info(msg)
            total_loss_gaze = 0.0
            num_train_batches = 0.0

            for one_batch in dataset:

                start_time = time.perf_counter()

                batch_loss_gaze = self.train_step(one_batch)
                total_loss_gaze += batch_loss_gaze
                num_train_batches += 1
                if num_train_batches % self.print_freq == 0:
                    msg = 'Trained batch: {batch}\t' \
                          'Batch loss: {batch_loss:.5f}\t' \
                          'Epoch total loss: {total_loss:.5f}\t' \
                          'Cost Time: {cost_time:.5f}\t' \
                          'Date: {date}'.format(
                        batch=num_train_batches,
                        batch_loss=batch_loss_gaze,
                        total_loss=total_loss_gaze,
                        cost_time=time.perf_counter() - start_time,
                        date=time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
                    )
                    logger.info(msg)

            return total_loss_gaze / num_train_batches

        def val_epoch(dataset):
            msg = 'Start validating...'
            logger.info(msg)
            total_loss_gaze = 0.0
            num_val_batches = 0.0

            for one_batch in dataset:

                start_time = time.perf_counter()

                batch_loss_gaze = self.val_step(one_batch)
                total_loss_gaze += batch_loss_gaze
                num_val_batches += 1
                if num_val_batches % self.print_freq == 0:
                    msg = 'Validated batch: {batch}\t' \
                          'Batch loss: {batch_loss:.5f}\t' \
                          'Epoch total loss: {total_loss:.5f}\t' \
                          'Cost Time: {cost_time:.5f}\t' \
                          'Date: {date}'.format(
                        batch=num_val_batches,
                        batch_loss=batch_loss_gaze,
                        total_loss=total_loss_gaze,
                        cost_time=time.perf_counter() - start_time,
                        date=time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
                    )
                    logger.info(msg)
            return total_loss_gaze / num_val_batches

        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.lr_decay()
            self.summary_writer_train.add_scalar('epoch learning rate', self.current_learning_rate, epoch)

            msg = 'Start epoch {} with learning rate {}'.format(epoch, self.current_learning_rate)
            logger.info(msg)

            train_loss_gaze = train_epoch(self.train_dataset)
            msg = 'Epoch {} train loss {}'.format(epoch, train_loss_gaze)
            logger.info(msg)
            self.summary_writer_train.add_scalar('epoch loss gaze', train_loss_gaze, epoch)

            val_loss_gaze = val_epoch(self.val_dataset)
            msg = 'Epoch {} val loss {}'.format(epoch, val_loss_gaze)
            logger.info(msg)
            self.summary_writer_val.add_scalar('epoch loss gaze', val_loss_gaze, epoch)

            # save model when reach a new lowest validation loss
            if val_loss_gaze < self.lowest_val_loss:
                if not os.path.exists(os.path.join('./models')):
                    os.makedirs(os.path.join('./models'))
                model_name = './models/model-{}-epoch-{}-loss-{:.5f}.pth'.format(self.version, epoch, val_loss_gaze)
                torch.save(self.model_frame, model_name)
                msg = f'Save model at: {model_name}'
                logger.info(msg)
                self.best_model = model_name
                self.lowest_val_loss = val_loss_gaze
            self.last_val_loss = val_loss_gaze

        return self.best_model
