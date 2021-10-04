from options.train_options import TrainOptions
from models import create_model
from util.visualizer import Visualizer
import torch
from tqdm import trange, tqdm
from fusion_dataset import *
from util import util


class MyTrainer():
    def __init__(self):
        self.dataset = self.choose_dataset()
        self.dataset_loader = torch.utils.data.DataLoader(self.dataset,
                                                          batch_size=opt.batch_size,
                                                          shuffle=True,
                                                          num_workers=8)
        self.dataset_size = len(self.dataset)
        print('#training images = %d' % self.dataset_size)
        self.model = create_model(opt)
        self.model.setup(opt)
        opt.display_port = 8098
        self.visualizer = Visualizer(opt)
        self.total_steps = 0

    def choose_dataset(self):
        dataset = None
        opt = TrainOptions().parse()
        if opt.stage == 'full':
            dataset = Training_Full_Dataset(opt)
        elif opt.stage == 'instance':
            dataset = Training_Instance_Dataset(opt)
        elif opt.stage == 'fusion':
            dataset = Training_Fusion_Dataset(opt)
        else:
            print('Error! Wrong stage selection!')
            exit()

        return dataset

    def display_result(self, epoch):
        if self.total_steps % opt.display_freq == 0:
            save_result = self.total_steps % opt.update_html_freq == 0
            self.visualizer.display_current_results(self.model.get_current_visuals(), epoch, save_result)

    def plot_loss(self, epoch, epoch_iter):
        if self.total_steps % opt.print_freq == 0:
            losses = self.model.get_current_losses()
            if opt.display_id > 0:
                self.visualizer.plot_current_losses(epoch, float(epoch_iter) / self.dataset_size, opt, losses)

    def iter_one_two(self, epoch, epoch_iter):
        for data_raw in tqdm(self.dataset_loader, desc='batch', dynamic_ncols=True, leave=False):
            self.total_steps += opt.batch_size
            epoch_iter += opt.batch_size

            data_raw['rgb_img'] = [data_raw['rgb_img']]
            data_raw['gray_img'] = [data_raw['gray_img']]

            input_data = util.get_colorization_data(data_raw['gray_img'], opt, p=1.0, ab_thresh=0)
            gt_data = util.get_colorization_data(data_raw['rgb_img'], opt, p=1.0, ab_thresh=10.0)
            if gt_data is None:
                continue
            if (gt_data['B'].shape[0] < opt.batch_size):
                continue
            input_data['B'] = gt_data['B']
            input_data['hint_B'] = gt_data['hint_B']
            input_data['mask_B'] = gt_data['mask_B']

            self.visualizer.reset()
            self.model.set_input(input_data)
            self.model.optimize_parameters()

            self.display_result(epoch)
            self.plot_loss(epoch, epoch_iter)

    def iter_three(self, epoch, epoch_iter):
        for data_raw in tqdm(self.dataset_loader, desc='batch', dynamic_ncols=True, leave=False):
            self.total_steps += opt.batch_size
            epoch_iter += opt.batch_size

            box_info = data_raw['box_info'][0]
            box_info_2x = data_raw['box_info_2x'][0]
            box_info_4x = data_raw['box_info_4x'][0]
            box_info_8x = data_raw['box_info_8x'][0]
            cropped_input_data = util.get_colorization_data(data_raw['cropped_gray'], opt, p=1.0, ab_thresh=0)
            cropped_gt_data = util.get_colorization_data(data_raw['cropped_rgb'], opt, p=1.0, ab_thresh=10.0)
            full_input_data = util.get_colorization_data(data_raw['full_gray'], opt, p=1.0, ab_thresh=0)
            full_gt_data = util.get_colorization_data(data_raw['full_rgb'], opt, p=1.0, ab_thresh=10.0)
            if cropped_gt_data is None or full_gt_data is None:
                continue
            cropped_input_data['B'] = cropped_gt_data['B']
            full_input_data['B'] = full_gt_data['B']
            ############################################

            self.visualizer.reset()
            self.model.set_input(cropped_input_data)
            self.model.set_fusion_input(full_input_data, [box_info, box_info_2x, box_info_4x, box_info_8x])
            self.model.optimize_parameters()

            self.display_result(epoch)
            self.plot_loss(epoch, epoch_iter)


    def train_one_two(self):

        for epoch in trange(opt.epoch_count, opt.niter + opt.niter_decay, desc='epoch', dynamic_ncols=True):
            epoch_iter = 0 # TODO: double check
            self.iter_one_two(epoch, epoch_iter) # TODO: double check

            # Save model
            if epoch % opt.save_epoch_freq == 0:
                self.model.save_networks('latest')
                self.model.save_networks(epoch)
            self.model.update_learning_rate()


    def train_three(self):

        for epoch in trange(opt.epoch_count, opt.niter + opt.niter_decay, desc='epoch', dynamic_ncols=True):
            epoch_iter = 0 # TODO: double check
            self.iter_three(epoch, epoch_iter) # TODO: double check

            # Save model
            if epoch % opt.save_epoch_freq == 0:
                self.model.save_fusion_epoch(epoch)
            self.model.update_learning_rate()

    def train(self):
        # train stage one or two
        if opt.stage == 'full' or opt.stage == 'instance':
            self.train_one_two()
        # train stage three
        elif opt.stage == 'fusion':
            self.train_three()
        else:
            print('Error! Wrong stage selection!')
            exit()



if __name__ == '__main__':
    t = MyTrainer()
    t.train()
