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

os.environ['MKL_THREADING_LAYER'] = 'GNU'

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
    s2=0
    for imagesinbatch in  range(gen.shape[0]):
        _psnr, _ssim,s2_ = utils.calc_psnr_and_ssim(real[imagesinbatch].detach(), gen[imagesinbatch].detach())
        psnr += _psnr
        ssim += _ssim
        s2=+s2_
    psnr=psnr/gen.shape[0]
    ssim=ssim/gen.shape[0]         
    s2=s2/gen.shape[0]         
    return psnr,ssim,s2
def nbnorm(x):
    return (x-x.min())/ (x.max()-x.min())
    
def testdataset(generator,datasetloc,_logger):
    datasetname=os.path.basename(datasetloc)
    data_loader,tot_batch = utils.get_dataloader(datasetloc,img_size=args.img_size, batch_size = args.batch_size)
    psnr=ssim=psnr2=ssim2=S2=0
    i=0
    with torch.no_grad():
        for data in tqdm(range(tot_batch)):
            #images, labels = data
            # calculate outputs by running images through the network
            realB4 = next(data_loader).to(device)
            realB2 = nn.functional.interpolate(realB4, scale_factor=.5, mode='bilinear',align_corners=True)
            realA = nn.functional.interpolate(realB2, scale_factor=.5, mode='bilinear',align_corners=True)
            
            realAup2 = nn.Upsample(scale_factor=2)(realB2)
            realAup4 = nn.Upsample(scale_factor=4)(realA)
            
            realA2 = nn.Upsample(scale_factor=2)(realA)
            gen4 = generator(realA2)
            gen2 = generator(realB2)
            #gen4= gen2=generator(realB2)
            # the class with the highest energy is what we choose as prediction
            psnr_,ssim_,S2_=calpsnrssim(gen2,realB4)
            psnr2_,ssim2_,S22_=calpsnrssim(gen4,realB4)
            #psnr2_,ssim2_=calpsnrssim(gen4,realB4)
            psnr+=psnr_
            psnr2+=psnr2_
            ssim+=ssim_
            ssim2+=ssim2_
            S2+=S2_
            #map(generator,realB2)
            
            nrow=int(math.sqrt( realA.shape[0]))
            realimg = T.ToPILImage()(make_grid(realA, nrow = nrow, padding = 10, normalize = True))
            groundth2 = T.ToPILImage()(make_grid(realB2, nrow = nrow, padding = 10, normalize = True))
            groundth4 = T.ToPILImage()(make_grid(realB4, nrow = nrow, padding = 10, normalize = True))
            genimg2 = T.ToPILImage()(make_grid(gen2, nrow = nrow, padding = 10, normalize = True))
            genimg4 = T.ToPILImage()(make_grid(gen4, nrow = nrow, padding = 10, normalize = True))
            get_concat_h([realimg,genimg4,groundth4]).save(str(args.weight)+'/test_results/'+datasetname+str(i)+'x4.jpg')
            get_concat_h([groundth2,genimg2,groundth4]).save(str(args.weight)+'/test_results/'+datasetname+str(i)+'x2.jpg')
            
            grounddiff4=T.functional.equalize(T.ToPILImage()(nbnorm(realB4-realAup4).squeeze()))            
            genrateddiff4=T.functional.equalize(T.ToPILImage()(nbnorm(gen4-realAup4).squeeze()))
            grounddiff2=T.functional.equalize(T.ToPILImage()(nbnorm(realB4-realAup2).squeeze()))
            genrateddiff2=T.functional.equalize(T.ToPILImage()(nbnorm(gen2-realAup2).squeeze()))
            
            plt.subplot(121)
            plt.imshow(grounddiff4,cmap='jet')
            plt.subplot(122)
            plt.imshow(genrateddiff4,cmap='jet')
            plt.subplots_adjust(bottom=0.0, right=0.8, top=1.0)
            cax = plt.axes([0.85, 0.1, 0.075, 0.8])
            plt.colorbar(cax=cax)
            plt.savefig(str(args.weight)+'/diff/'+datasetname+str(i)+'x4.jpg')
            plt.close('all')
            
            plt.subplot(121)
            plt.imshow(grounddiff2,cmap='jet')
            plt.subplot(122)
            plt.imshow(genrateddiff2,cmap='jet')
            plt.subplots_adjust(bottom=0.0, right=0.8, top=1.0)
            cax = plt.axes([0.85, 0.1, 0.075, 0.8])
            plt.colorbar(cax=cax)
            plt.savefig(str(args.weight)+'/diff/'+datasetname+str(i)+'x2.jpg')
            plt.close('all')
            
            #break
            i=i+1
            
    psnr=psnr/tot_batch
    psnr2=psnr2/tot_batch
    ssim=ssim/tot_batch
    ssim2=ssim2/tot_batch
    S2=S2/tot_batch
    _logger.info(datasetname+': x2psnr/ssim: %.3f' %psnr+'/%.3f' %ssim+' x4psnr/ssim: %.3f' %psnr2+'/%.3f' %ssim2)
    print('psnrx2,ssimx2,psnr4,ssim4',psnr,ssim,psnr2,ssim2,S2)
    
def test(generator):
    _logger=utils.Logger(log_file_name=os.path.join(str(args.weight),'test.log'), 
        logger_name='testlog').get_log()
    _logger.info("test module")
    generator.eval()
    #testdataset(generator,'../dataset/DIV2K_valid_HR',_logger )
    testdataset(generator,'../dataset/Set5',_logger )
    testdataset(generator,'../dataset/Set14',_logger )
    testdataset(generator,'../dataset/BSD100',_logger )
    # testdataset(generator,'../dataset/urban100',_logger )
    # testdataset(generator,'../dataset/manga',_logger )
    #testdataset(generator,'../dataset/FFHQtest',_logger )
    
    
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

    
    #device = torch.device("cuda:0")
    # Initialize Generator and Discriminator
    netG = nn.DataParallel(model4.Generator()).to(device)
    netG.load_state_dict(copy.deepcopy(torch.load(str(args.weight)+"/weights/Generator.pth")))   
    # Loss function
    criterion = nn.BCELoss()
    criterionL1 = torch.nn.L1Loss()
    criterionL2 = torch.nn.MSELoss()
    
    # Start Testing
    test(netG)
