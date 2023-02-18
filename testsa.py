#!python test41.py --batch-size 1 --img_size 256 --weight '/workspace/drive/neeraj/nb/vitgan3/save_dir'

import numpy
import os
import argparse
import copy
import math 
import utils
import torch
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import Restormer as models
import model4
import matplotlib.pyplot as plt
import matplotlib
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.autograd import Variable

import matplotlib.cm as mpl_color_map
import cv2


def get_concat_h(im):
    #print('im',len(im))
    iw=0
    for i in im:
        iw=iw+i.width
    dst = Image.new('RGB', (iw, im[len(im)-1].height))
    
    iw=0
    for i in im:
        dst.paste(i, (iw, 0))
        iw=iw+i.width
    return dst

def calpsnrssim(gen,real):
    psnr=0
    ssim=0
    for imagesinbatch in  range(gen.shape[0]):
        _psnr, _ssim = utils.calc_psnr_and_ssim(real[imagesinbatch].detach(), gen[imagesinbatch].detach())
        psnr += _psnr
        ssim += _ssim
    psnr=psnr/gen.shape[0]
    ssim=ssim/gen.shape[0]         
    return psnr,ssim
from torch.nn.functional import normalize
def mapdataset(generator,datasetloc):
    generator.eval()
    datasetname=os.path.basename(datasetloc)
    i=0
    data_loader,tot_batch = utils.get_dataloader(datasetloc,img_size=args.img_size, batch_size = args.batch_size)
    for data in tqdm(range(tot_batch)):
        realB2 = next(data_loader).to(device)
        realA = nn.functional.interpolate(realB2, scale_factor=.5, mode='bicubic',align_corners=True)
        real_A = Variable(realA).cuda()
        
        A= real_A.requires_grad_()
        fake_B = generator.forward(A)
        #temp = torch.ones(1, 3, fake_B.shape[3], fake_B.shape[3], dtype=torch.float, requires_grad=True, device='cuda')
        
        fake_B.backward(realB2)
        saliency, _ = torch.max(A.grad.data.abs(), dim=1)
        saliency=saliency.cpu()
        
        saliency=(saliency-saliency.min())/ (saliency.max()-saliency.min())
        
        saliency=T.ToPILImage()(saliency) 
        saliency=T.functional.equalize(saliency)
        saliency = T.Compose([T.PILToTensor()])(saliency)
        saliency=(saliency-saliency.min())/ (saliency.max()-saliency.min())
        saliency=saliency.squeeze()
        imageA = A.squeeze()
        imageA=(imageA+1)/2
        
        plt.subplot(121)
        plt.imshow(imageA.cpu().detach().numpy().transpose(1,2,0))
        plt.subplot(122)
        plt.imshow(saliency,cmap='jet')
        plt.subplots_adjust(bottom=0.0, right=0.8, top=1.0)
        cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        plt.colorbar(cax=cax)
        plt.savefig(str(args.weight)+'/saliency/'+datasetname+str(i)+'a.png')
        plt.close('all')
        i=i+1
        
        
def map(generator):
    _logger=utils.Logger(log_file_name=os.path.join(str(args.weight),'test.log'), 
        logger_name='testlog').get_log()
    _logger.info("test module")
    generator.eval()
    #mapdataset(generator,'../dataset/DIV2K_valid_HR' )
    mapdataset(generator,'../dataset/Set5' )
    mapdataset(generator,'../dataset/Set14' )
    mapdataset(generator,'../dataset/BSD100' )
    # mapdataset(generator,'../dataset/urban100' )
    # mapdataset(generator,'../dataset/manga' )
    #mapdataset(generator,'../dataset/FFHQtest')
    generator.train()
    return 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type = int, default = 1,
                        help = "Size of each batches (Default: 1)")
    parser.add_argument("--beta1", type = float, default = 0.0,
                        help = "Coefficients used for computing running averages of gradient and its square")
    parser.add_argument("--beta2", type = float, default = 0.99,
                        help = "Coefficients used for computing running averages of gradient and its square")
    parser.add_argument("--data-dir", type = str, default = "dataset/cufed_small/train/input",
                        help = "Data root dir of your training data")
    parser.add_argument("--img_size", type = int, default = 512,
                        help = "Select the specific img_size to training")
    parser.add_argument("--weight", type = str, default = "",
                        help = "Select the specific img_size to training")
    parser.add_argument("--gpuid", type = str, default = 0,
                        help = "Select the specific gpu to training")
    args = parser.parse_args()
    # Device
    # Device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize Generator and Discriminator
    netG = nn.DataParallel(model4.Generator()).to(device)
    netG.load_state_dict(copy.deepcopy(torch.load(str(args.weight)+"/weights/Generator.pth")))   
    # Loss function
    criterion = nn.BCELoss()
    criterionL1 = torch.nn.L1Loss()
    criterionL2 = torch.nn.MSELoss()
    map(netG)
