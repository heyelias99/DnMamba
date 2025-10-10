import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from prep import printProgressBar
from networks import RED_CNN
from measure import compute_measure
from measure import compute_SSIM
from measure import compute_MSE
from DnMamba import DnMamba
from torch.optim.lr_scheduler import MultiStepLR
import imageio

def mse_ssim_loss(pred,y):
    return 0.70*compute_MSE(pred,y)+0.30*(1-compute_SSIM(pred,y))

class Solver(object):
    def __init__(self, args, data_loader):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig

        self.patch_size = args.patch_size
        
        # self.model = RED_CNN()


        self.model = DnMamba(img_size=(64, 64), 
                   window_size=8, img_range=1.,depths=[6, 6, 6, 6,6,6],
                   embed_dim=32)
        
        
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        self.lr = args.lr
        # self.criterion = mse_ssim_loss()
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,milestones=[40,80],gamma=0.5)


    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
        torch.save(self.model.state_dict(), f)


    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
        print(f)
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f).items():
                n = k[7:]
                state_d[n] = v
            self.model.load_state_dict(state_d)
        else:
            self.model.load_state_dict(torch.load(f))


    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image


    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat


    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close()
    
    def save_figs(self, x, y, pred, figpath_x, figpath_y, figpath_pred, fig_name):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()

        plt.imsave(os.path.join(figpath_x, 'x_{}.png'.format(fig_name)), x,cmap='gray')
        plt.imsave(os.path.join(figpath_y, 'y_{}.png'.format(fig_name)), y,cmap='gray')
        plt.imsave(os.path.join(figpath_pred, 'pred_{}.png'.format(fig_name)), pred,cmap='gray')


    def train(self):
        train_losses = []
        total_iters = 0
        start_time = time.time()
        

        for epoch in range(1, self.num_epochs):
            self.model.train(True)

            for iter_, (x, y) in enumerate(self.data_loader):
                total_iters += 1

                # add 1 channel
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                if self.patch_size: # patch training
                    x = x.view(-1, 1, self.patch_size, self.patch_size)
                    y = y.view(-1, 1, self.patch_size, self.patch_size)

                pred = self.model(x)
                # loss = self.criterion(pred, y)
                loss = mse_ssim_loss(pred,y)
                self.model.zero_grad()
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

                # print
                if total_iters % self.print_iters == 0:
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(total_iters, epoch, 
                                                                                                        self.num_epochs, iter_+1, 
                                                                                                        len(self.data_loader), loss.item(), 
                                                                                                        time.time() - start_time))
                # learning rate decay
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()
                # save model
                if total_iters % self.save_iters == 0:
                    self.save_model(total_iters)
                    np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))
            self.scheduler.step()
            print(epoch, self.scheduler.get_lr())

    def test(self):
        del self.model
        # load
        self.REDCNN = RED_CNN().to(self.device)

        
        self.model = DnMamba(img_size=(64, 64), 
                   window_size=8, img_range=1.,depths=[6, 6, 6, 6,6,6],
                   embed_dim=32).cuda()
 
        self.load_model(self.test_iters)

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
        ori_psnr_avg_array = np.array([])
        ori_ssim_avg_array = np.array([])
        ori_rmse_avg_array = np.array([])
        pred_psnr_avg_array = np.array([])
        pred_ssim_avg_array = np.array([])
        pred_rmse_avg_array = np.array([])
        with torch.no_grad():
            for i, (x, y) in enumerate(self.data_loader):
                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                pred = self.model(x)

                # denormalize, truncate
                # x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                # y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                # pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))
                x = self.denormalize_(x.view(shape_, shape_).cpu().detach())
                y = self.denormalize_(y.view(shape_, shape_).cpu().detach())
                pred = self.denormalize_(pred.view(shape_, shape_).cpu().detach())

                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)

                ori_psnr_avg_array = np.append(ori_psnr_avg_array,original_result[0])
                ori_ssim_avg_array = np.append(ori_ssim_avg_array,original_result[1])
                ori_rmse_avg_array = np.append(ori_rmse_avg_array,original_result[2])
            
                pred_psnr_avg_array = np.append(pred_psnr_avg_array,pred_result[0])
                pred_ssim_avg_array = np.append(pred_ssim_avg_array,pred_result[1])
                pred_rmse_avg_array = np.append(pred_rmse_avg_array,pred_result[2])
                
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]


                # save result figure
                if self.result_fig:
                    self.save_fig(x, y, pred, i, original_result, pred_result)
                    figpath_x = './save_10_DnMamba_without_ca/figs/x/'
                    figpath_y = './save_10_DnMamba_without_ca/figs/y/'
                    figpath_pred = './save_10_DnMamba_without_ca/figs/pred/'
                    self.save_figs( x, y, pred, figpath_x, figpath_y, figpath_pred, i)

                printProgressBar(i, len(self.data_loader),
                                 prefix="Compute measurements ..",
                                 suffix='Complete', length=25)
                
            np.savetxt(os.path.join(self.save_path, 'psnr.csv'),pred_psnr_avg_array, delimiter=',')
            np.savetxt(os.path.join(self.save_path, 'ssim.csv'),pred_ssim_avg_array, delimiter=',')
            np.savetxt(os.path.join(self.save_path, 'rmse.csv'),pred_rmse_avg_array, delimiter=',')
                

            print(ori_psnr_avg_array.std(),ori_ssim_avg_array.std(),ori_rmse_avg_array.std(),pred_psnr_avg_array.std(),pred_ssim_avg_array.std(),pred_rmse_avg_array.std())
            print('\n')
            print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(self.data_loader), 
                                                                                            ori_ssim_avg/len(self.data_loader), 
                                                                                            ori_rmse_avg/len(self.data_loader)))
            print('\n')
            print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(self.data_loader), 
                                                                                                  pred_ssim_avg/len(self.data_loader), 
                                                                                                  pred_rmse_avg/len(self.data_loader)))
