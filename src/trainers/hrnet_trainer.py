import os
import math
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HRNetTrainer(object):
    def __init__(self,
                 model,
                 train_dataset,
                 val_dataset,
                 epochs=100,
                 initial_learning_rate=0.001,
                 start_epoch=1,
                 print_freq=500,
                 version='v0.1',
                 tensorboard_dir='./logs'):
        super(HRNetTrainer, self).__init__()
        self.version = version
        self.model = model
        self.print_freq = print_freq

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.epochs = epochs
        self.current_learning_rate = initial_learning_rate
        self.start_epoch = start_epoch

        self.loss_obj = nn.MSELoss(reduction='mean')
        self.optimizer = self.optimizer = optim.Adam(self.model.parameters(), lr=self.current_learning_rate, weight_decay=1e-4)
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.current_learning_rate, weight_decay=1e-4)

    def compute_coord_loss(self, predict, label):
        loss = self.loss_obj(predict, label)
        return loss

    def train_step(self, inputs):
        eye_input = inputs['eye'].cuda()
        heatmaps_label = inputs['heatmaps'].cuda()
        ldmks_label = inputs['landmarks'].cuda()
        radius_label = inputs['radius'].cuda()

        heatmaps_predict, ldmks_predict, radius_predict = self.model(eye_input)
        loss_heatmaps = 0.1 * self.compute_coord_loss(heatmaps_label, heatmaps_predict)
        loss_ldmks = self.compute_coord_loss(ldmks_predict, ldmks_label)
        loss_radius = 0.01 * self.compute_coord_loss(radius_predict, torch.unsqueeze(radius_label, dim=-1))

        loss = loss_heatmaps + loss_ldmks + loss_radius
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_heatmaps.item(), loss_ldmks.item(), loss_radius.item()

    def val_step(self, inputs):
        eye_input = inputs['eye'].cuda()
        heatmaps_label = inputs['heatmaps'].cuda()
        ldmks_label = inputs['landmarks'].cuda()
        radius_label = inputs['radius'].cuda()

        heatmaps_predict, ldmks_predict, radius_predict = self.model(eye_input)
        loss_heatmaps = 0.1 * self.compute_coord_loss(heatmaps_label, heatmaps_predict)
        loss_ldmks = self.compute_coord_loss(ldmks_predict, ldmks_label)
        loss_radius = 0.01 * self.compute_coord_loss(radius_predict, torch.unsqueeze(radius_label, dim=-1))
        return loss_heatmaps.item(), loss_ldmks.item(), loss_radius.item()

    def run(self):
        def train_epoch(dataset):
            msg = 'Start training...'
            logger.info(msg)
            total_loss_heatmaps = 0.0
            total_loss_ldmks = 0.0
            total_loss_radius = 0.0
            num_train_batches = 0.0

            for one_batch in dataset:
                
                start_time = time.perf_counter()

                batch_loss_heatmaps, batch_loss_ldmks, batch_loss_radius = self.train_step(one_batch)
                total_loss_heatmaps += batch_loss_heatmaps
                total_loss_ldmks += batch_loss_ldmks
                total_loss_radius += batch_loss_radius
                num_train_batches += 1
                if num_train_batches % self.print_freq == 0:
                    msg = 'Trained batch: {batch}\t' \
                          'Batch loss: {batch_loss:.5f}\t' \
                          'Epoch total loss: {total_loss:.5f}\t' \
                          'Cost Time: {cost_time:.5f}\t' \
                          'Date: {date}'.format(
                                batch=num_train_batches,
                                batch_loss=batch_loss_heatmaps + batch_loss_ldmks + batch_loss_radius,
                                total_loss=total_loss_heatmaps + total_loss_ldmks + total_loss_radius,
                                cost_time=time.perf_counter() - start_time,
                                date=time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
                                )
                    logger.info(msg)

            return total_loss_heatmaps / num_train_batches,\
                   total_loss_ldmks / num_train_batches, \
                   total_loss_radius / num_train_batches

        def val_epoch(dataset):
            msg = 'Start validating...'
            logger.info(msg)
            total_loss_heatmaps = 0.0
            total_loss_ldmks = 0.0
            total_loss_radius = 0.0
            num_val_batches = 0.0
            for one_batch in dataset:

                start_time = time.perf_counter()

                batch_loss_heatmaps, batch_loss_ldmks, batch_loss_radius = self.val_step(one_batch)
                total_loss_heatmaps += batch_loss_heatmaps
                total_loss_ldmks += batch_loss_ldmks
                total_loss_radius += batch_loss_radius
                num_val_batches += 1
                if num_val_batches % self.print_freq == 0:
                    msg = 'Trained batch: {batch}\t' \
                          'Batch loss: {batch_loss:.5f}\t' \
                          'Epoch total loss: {total_loss:.5f}\t' \
                          'Cost Time: {cost_time:.5f}\t' \
                          'Date: {date}'.format(
                                batch=num_val_batches,
                                batch_loss=batch_loss_heatmaps + batch_loss_ldmks + batch_loss_radius,
                                total_loss=total_loss_heatmaps + total_loss_ldmks + total_loss_radius,
                                cost_time=time.perf_counter() - start_time,
                                date=time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
                                )
                    logger.info(msg)
            return total_loss_heatmaps / num_val_batches,\
                   total_loss_ldmks / num_val_batches, \
                   total_loss_radius / num_val_batches

        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.lr_decay()
            self.summary_writer_train.add_scalar('epoch learning rate', self.current_learning_rate, epoch)

            msg = 'Start epoch {} with learning rate {}'.format(epoch, self.current_learning_rate)
            logger.info(msg)

            train_loss_heatmaps, train_loss_ldmks, train_loss_radius = train_epoch(self.train_dataset)
            total_train_loss = train_loss_heatmaps + train_loss_ldmks + train_loss_radius
            msg = 'Epoch {} train loss {}'.format(epoch, total_train_loss)
            logger.info(msg)
            self.summary_writer_train.add_scalar('epoch loss', total_train_loss, epoch)
            self.summary_writer_train.add_scalar('epoch loss heatmaps', 10 * train_loss_heatmaps, epoch)
            self.summary_writer_train.add_scalar('epoch loss ldmks', train_loss_ldmks, epoch)
            self.summary_writer_train.add_scalar('epoch loss radius', 100 * train_loss_radius, epoch)

            val_loss_heatmaps, val_loss_ldmks, val_loss_radius = val_epoch(self.val_dataset)
            total_val_loss = val_loss_heatmaps + val_loss_ldmks + val_loss_radius
            msg = 'Epoch {} val loss {}'.format(epoch, total_val_loss)
            logger.info(msg)
            self.summary_writer_val.add_scalar('epoch loss', total_val_loss, epoch)
            self.summary_writer_val.add_scalar('epoch loss heatmaps', 10 * val_loss_heatmaps, epoch)
            self.summary_writer_val.add_scalar('epoch loss ldmks', val_loss_ldmks, epoch)
            self.summary_writer_val.add_scalar('epoch loss radius', 100 * val_loss_radius, epoch)

            # save model when reach a new lowest validation loss
            if total_val_loss < self.lowest_val_loss:
                if not os.path.exists(os.path.join('./models')):
                    os.makedirs(os.path.join('./models'))
                model_name = './models/model-{}-epoch-{}-loss-{:.5f}.pth'.format(self.version, epoch, total_val_loss)
                torch.save(self.model, model_name)
                msg = f'Save model at: {model_name}'
                logger.info(msg)
                self.best_model = model_name
                self.lowest_val_loss = total_val_loss
            self.last_val_loss = total_val_loss

        return self.best_model
