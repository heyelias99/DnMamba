import torch
from loader import get_loader
from measure import compute_measure
import os
import numpy as np
import pandas as pd
from networks import RED_CNN
from Dnmamba import DnMamba
def trunc(mat):
    mat[mat <= -160] = -160
    mat[mat >= 240] = 240
    return mat
    
def denormalize_(image):
    image = image * (3072.0 - (-1024.0)) + (-1024.0)
    return image

def test(model_path):


    
    model = RED_CNN().cuda()
    # model = DnMamba(img_size=(64, 64), 
    #                window_size=8, img_range=1.,depths=[6, 6, 6, 6],
    #                embed_dim=32).cuda()
    
    model.load_state_dict(torch.load(model_path))

    # compute PSNR, SSIM, RMSE
    ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
    pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

    data_loader = get_loader(mode='test',
                             load_mode=0,
                             saved_path='10_npy_img',
                             test_patient='L056',
                             patch_n=None,
                             patch_size=None,
                             transform=False,
                             batch_size=1,
                             num_workers=7)
    
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            shape_ = x.shape[-1]
            x = x.unsqueeze(0).float().cuda()
            y = y.unsqueeze(0).float().cuda()

            pred = model(x)

            # denormalize, truncate
            x = trunc(denormalize_(x.view(shape_, shape_).cpu().detach()))
            y = trunc(denormalize_(y.view(shape_, shape_).cpu().detach()))
            pred = trunc(denormalize_(pred.view(shape_, shape_).cpu().detach()))

            data_range = 240 - (-160)

            original_result, pred_result = compute_measure(x, y, pred, data_range)
            ori_psnr_avg += original_result[0]
            ori_ssim_avg += original_result[1]
            ori_rmse_avg += original_result[2]
            pred_psnr_avg += pred_result[0]
            pred_ssim_avg += pred_result[1]
            pred_rmse_avg += pred_result[2]
            # print(pred_result[0])
            # print(pred_result[2])

        # ori_psnr_avg/len(data_loader), ori_ssim_avg/len(data_loader), ori_rmse_avg/len(data_loader)
        result = pred_psnr_avg/len(data_loader), pred_ssim_avg/len(data_loader), pred_rmse_avg/len(data_loader)
        return result


if __name__ == "__main__":
    folder_path = '/root/autodl-tmp/DNM/save_10_REDCNN'

    psnr = np.array([])
    ssim = np.array([])
    rmse = np.array([])
    

    for i in range(40,109):
        path = folder_path+"/REDCNN_{}000iter.ckpt".format(i)
        psnr = np.append(psnr,test(path)[0])
        ssim = np.append(ssim,test(path)[1])
        rmse = np.append(rmse,test(path)[2])
        print(test(path))
        print('psnr',psnr)
        print('ssim',ssim)
        print('rmse',rmse)

        print(path)
    
        allp = [psnr,ssim,rmse]
    
        np.savetxt('data_DNM_con.csv',allp, delimiter=',')
