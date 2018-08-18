from __future__ import print_function

import os
from glob import glob
from tqdm import trange
from itertools import chain

import torch
from torch import nn
import torch.nn.parallel
import torchvision.utils as vutils
from torch.autograd import Variable

from models import *
from data_loader import get_loader
import evaluate
from tensorboardX import SummaryWriter
from img_random_discmp import img_random_dis 
import numpy as np
import time


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
def get_psnr(var_tar, var_pred):
    return evaluate.get_psnr(
        var_tar.data.mul(255).round().permute(0, 2, 3, 1).cpu().numpy().clip(0, 255),
        var_pred.data.mul(255).round().permute(0, 2, 3, 1).cpu().numpy().clip(0, 255))
def criteriond(input1, input2):
        # Real
        err = torch.mean(torch.mul((input1 - input2),(input1 - input2)))
        #err = torch.mean(torch.abs(input1 - input2))
        return err
def tensor_rgb2gray(valid_x_A):
    x_A_t2a = valid_x_A.numpy()
    #x_B_t2a=x_B.numpy()
    x_A_t2a, x_B_t2a = img_random_dis(x_A_t2a)
    valid_x_A=torch.from_numpy(x_A_t2a)
    valid_x_B=torch.from_numpy(x_B_t2a)
    valid_x_A=valid_x_A.float()
    valid_x_B=valid_x_B.float()
    return valid_x_A, valid_x_B
class Trainer(object):
    def __init__(self, config, a_data_loader):
        self.config = config

        self.a_data_loader = a_data_loader

        self.num_gpu = config.num_gpu
        self.dataset = config.dataset

        self.loss = config.loss
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.weight_decay = config.weight_decay

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.is_trainwithGAN = config.is_trainwithGAN

        self.build_model()

        if self.num_gpu == 1:
            self.G_AB.cuda()
            self.D_A.cuda()

        elif self.num_gpu > 1:
            self.G_AB = nn.DataParallel(self.G_AB.cuda(),device_ids=range(self.num_gpu))
            self.D_A = nn.DataParallel(self.D_A.cuda(),device_ids=range(self.num_gpu))

        if self.load_path:
            self.load_model()

    def build_model(self):
       
        a_channel = 1
        b_channel = 1

        
        self.G_AB = SGENGenerator(
                b_channel, a_channel)
        self.D_A = CommonDiscriminator(a_channel)

        self.G_AB.apply(weights_init)

        self.D_A.apply(weights_init)
    def load_model(self):
        print("[*] Load models from {}...".format(self.load_path))

        paths = glob(os.path.join(self.load_path, 'G_AB_*.pth'))
        paths.sort()

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.load_path))
            return

        idxes = [int(os.path.basename(path.split('.')[0].split('_')[-1])) for path in paths]
        self.start_step = max(idxes)

        if self.num_gpu == 0:
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        G_AB_filename = '{}/G_AB_{}.pth'.format(self.load_path, self.start_step)
        self.G_AB.load_state_dict(torch.load(G_AB_filename, map_location=map_location))
        if self.is_trainwithGAN:
            self.D_A.load_state_dict(
            torch.load('{}/D_A_{}.pth'.format(self.load_path, self.start_step), map_location=map_location))

       

        print("[*] Model loaded: {}".format(G_AB_filename))

    def train(self):
        writer = SummaryWriter()
        d = nn.MSELoss()
        bce = nn.BCELoss()

        real_label = 1
        fake_label = 0

        real_tensor = Variable(torch.FloatTensor(self.batch_size))
        _ = real_tensor.data.fill_(real_label)

        fake_tensor = Variable(torch.FloatTensor(self.batch_size))
        _ = fake_tensor.data.fill_(fake_label)

        if self.num_gpu > 0:
            d.cuda()
            bce.cuda()

            real_tensor = real_tensor.cuda()
            fake_tensor = fake_tensor.cuda()

        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

        optimizer_d = optimizer(
            chain(self.D_A.parameters()),
            lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        optimizer_g = optimizer(
            chain(self.G_AB.parameters()),
            lr=self.lr, betas=(self.beta1, self.beta2))

        A_loader = iter(self.a_data_loader)
        valid_x_A = A_loader.next()
        
        valid_x_A, valid_x_B = tensor_rgb2gray(valid_x_A)
        valid_x_A, valid_x_B = self._get_variable(valid_x_A), self._get_variable(valid_x_B)
        vutils.save_image(valid_x_A.data*0.5+0.5, '{}/valid_x_A.png'.format(self.model_dir))
        vutils.save_image(valid_x_B.data*0.5+0.5, '{}/valid_x_B.png'.format(self.model_dir))
        l_g_curt = Variable(torch.FloatTensor([2]), requires_grad=True)
        l_g_prev = Variable(torch.FloatTensor([0]), requires_grad=False)
        for step in trange(self.start_step, self.max_step):
            
            # update G network
            for i in range(1):
                
                try:
                    x_A = A_loader.next()
                    x_A, x_B = tensor_rgb2gray(x_A)
       
                except StopIteration:
                    A_loader = iter(self.a_data_loader)
                    x_A = A_loader.next()
                    x_A, x_B = tensor_rgb2gray(x_A)
                if x_A.size(0) != x_B.size(0):
                    print("[!] Sampled dataset from A and B have different # of data. Try resampling...")
                    continue

                x_A, x_B = self._get_variable(x_A), self._get_variable(x_B)

                batch_size = x_A.size(0)
                real_tensor.data.resize_(batch_size).fill_(real_label)
                fake_tensor.data.resize_(batch_size).fill_(fake_label)
                
                self.G_AB.zero_grad()

                x_AB = self.G_AB(x_A)
                l_const_B = criteriond(x_AB,x_B)

                l_g = l_const_B

                l_g.backward()
                optimizer_g.step()
                
            

            if step % self.log_step == 0:
                
                
                print("[{}/{}] l_const_B: {:.4f}". \
                      format(step, self.max_step, l_const_B.data[0]))
                psnr = get_psnr(x_AB*0.5+0.5, x_B*0.5+0.5)
                
                print("psnr: {:.4f} ".format(psnr)) 
                

            if step % self.save_step == self.save_step - 1:
                self.generate_with_A(x_A, x_B, self.model_dir, idx=step)
                print("[*] Save models to {}...".format(self.model_dir))

                torch.save(self.G_AB.state_dict(), '{}/G_AB_{}.pth'.format(self.model_dir, step))
    def train_withGAN(self):
        writer = SummaryWriter()
        d = nn.MSELoss()
        bce = nn.BCELoss()

        real_label = 1
        fake_label = 0

        real_tensor = Variable(torch.FloatTensor(self.batch_size))
        _ = real_tensor.data.fill_(real_label)

        fake_tensor = Variable(torch.FloatTensor(self.batch_size))
        _ = fake_tensor.data.fill_(fake_label)

        if self.num_gpu > 0:
            d.cuda()
            bce.cuda()

            real_tensor = real_tensor.cuda()
            fake_tensor = fake_tensor.cuda()

        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

        optimizer_d = optimizer(
            chain(self.D_A.parameters()),
            lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        optimizer_g = optimizer(
            chain(self.G_AB.parameters()),
            lr=self.lr, betas=(self.beta1, self.beta2))

        A_loader = iter(self.a_data_loader)
        valid_x_A = A_loader.next()
        
        valid_x_A, valid_x_B = tensor_rgb2gray(valid_x_A)
        valid_x_A, valid_x_B = self._get_variable(valid_x_A), self._get_variable(valid_x_B)
        vutils.save_image(valid_x_A.data*0.5+0.5, '{}/valid_x_A.png'.format(self.model_dir))
        vutils.save_image(valid_x_B.data*0.5+0.5, '{}/valid_x_B.png'.format(self.model_dir))
        l_g_curt = Variable(torch.FloatTensor([2]), requires_grad=True)
        l_g_prev = Variable(torch.FloatTensor([0]), requires_grad=False)
        for step in trange(self.start_step, self.max_step):  
            # update G network
            for i in range(1):
                try:
                    x_A = A_loader.next()
                    x_A, x_B = tensor_rgb2gray(x_A)
       
                except StopIteration:
                    A_loader = iter(self.a_data_loader)
                    x_A = A_loader.next()
                    x_A, x_B = tensor_rgb2gray(x_A)
                if x_A.size(0) != x_B.size(0):
                    print("[!] Sampled dataset from A and B have different # of data. Try resampling...")
                    continue

                x_A, x_B = self._get_variable(x_A), self._get_variable(x_B)

                batch_size = x_A.size(0)
                real_tensor.data.resize_(batch_size).fill_(real_label)
                fake_tensor.data.resize_(batch_size).fill_(fake_label)
                
                self.G_AB.zero_grad()

                x_AB = self.G_AB(x_A)
                l_const_A = criteriond(x_AB,x_B)
                l_gan_A = bce(self.D_A(x_AB), real_tensor)
                l_g = l_const_A + 0.1*l_gan_A

                l_g.backward()
                optimizer_g.step()
            # update D network
            self.D_A.zero_grad()

            x_AB = self.G_AB(x_A).detach()
            l_d_A_real, l_d_A_fake = bce(self.D_A(x_B), real_tensor), bce(self.D_A(x_AB), fake_tensor)
            
            l_d = l_d_A_real + l_d_A_fake
            l_d.backward()
            optimizer_d.step()
            

            

            if step % self.log_step == 0:
                
                print("[{}/{}] Loss_D: {:.4f} Loss_G: {:.4f}". \
                      format(step, self.max_step, l_d.data[0], l_g.data[0]))
                print("[{}/{}] l_const_A: {:.4f} l_gan_A: {:.4f}". \
                      format(step, self.max_step, l_const_A.data[0], l_gan_A.data[0]))
                psnr = get_psnr(x_AB*0.5+0.5, x_B*0.5+0.5)
                print("psnr: {:.4f} ".format(psnr)) 
            if step % self.save_step == self.save_step - 1:
                self.generate_with_A(x_A, x_B, self.model_dir, idx=step)
                print("[*] Save models to {}...".format(self.model_dir))

                torch.save(self.G_AB.state_dict(), '{}/G_AB_{}.pth'.format(self.model_dir, step))

                torch.save(self.D_A.state_dict(), '{}/D_A_{}.pth'.format(self.model_dir, step))
               

    def generate_with_A(self, inputs, inputs2, path, idx=None):
        x_AB = self.G_AB(inputs)
        x_BB = self.G_AB(inputs2)
        #x_ABA = self.G_BA(x_AB)

        x_AB_path = '{}/{}_x_AB.png'.format(path, idx)
        x_B_path = '{}/{}_x_B.png'.format(path, idx)
        x_A_path = '{}/{}_x_A.png'.format(path, idx)

        vutils.save_image(x_AB.data*0.5+0.5, x_AB_path)
        print("[*] Samples saved: {}".format(x_AB_path))
        vutils.save_image(inputs2.data*0.5+0.5, x_B_path)
        print("[*] Samples saved: {}".format(x_B_path))
        vutils.save_image(inputs.data*0.5+0.5, x_A_path)
        print("[*] Samples saved: {}".format(x_A_path))
           



    def test_bk(self):
        batch_size = self.config.sample_per_image
        A_loader = iter(self.a_data_loader)
        x_A = A_loader.next()
        x_A_t2a = x_A.numpy()
        N=30
        step = 0
        psnr_sum = np.zeros((N,))
        test_ind = 0
        test_dir = os.path.join(self.model_dir, 'test')
        time_start=time.time()

        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        for i in range(N):
            test_ind = i
            x_A_t, x_B_t = img_random_dis(x_A_t2a, test_ind)
            x_A=torch.from_numpy(x_A_t)
            x_B=torch.from_numpy(x_B_t)
            x_A=x_A.float()
            x_B=x_B.float()
            x_A, x_B = self._get_variable(x_A), self._get_variable(x_B)
            x_AB = self.G_AB(x_A)

            psnr = get_psnr(x_AB*0.5+0.5, x_B*0.5+0.5)
            psnr_sum[test_ind] = psnr_sum[test_ind] + psnr
            print("psnr: {:.4f} ".format(psnr)) 
            x_AB_path = '{}/{}_x_AB.png'.format(test_dir, step)
            x_B_path = '{}/{}_x_B.png'.format(test_dir, step)
            x_A_path = '{}/{}_x_A.png'.format(test_dir, step)
            
            vutils.save_image(x_AB.data*0.5+0.5, x_AB_path)
            print("[*] Samples saved: {}".format(x_AB_path))
            vutils.save_image(x_B.data*0.5+0.5, x_B_path)
            print("[*] Samples saved: {}".format(x_B_path))
            vutils.save_image(x_A.data*0.5+0.5, x_A_path)
            print("[*] Samples saved: {}".format(x_A_path))
            step += 1
            print(psnr_sum)
        
    def test(self):
        batch_size = self.config.sample_per_image
        A_loader = iter(self.a_data_loader)
        N=6
        step = np.zeros((N,))
        psnr_sum = np.zeros((N,))
        test_ind = 0
        test_dir = os.path.join(self.model_dir, 'test')
        time_start=time.time()

        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        while True:
            try:
                x_A = A_loader.next()
                x_A_t2a = x_A.numpy()
                #x_B_t2a=x_B.numpy()
                test_ind = (test_ind+1)%N
                x_A_t2a, x_B_t2a = img_random_dis(x_A_t2a, test_ind)
                x_A=torch.from_numpy(x_A_t2a)
                x_B=torch.from_numpy(x_B_t2a)
                x_A=x_A.float()
                x_B=x_B.float()
                x_A, x_B = self._get_variable(x_A), self._get_variable(x_B)
   
            except StopIteration:
                print("[!] Test sample restoration finished.")
                break
            
            x_AB = self.G_AB(x_A)

            psnr = get_psnr(x_AB*0.5+0.5, x_B*0.5+0.5)
            psnr_sum[test_ind] = psnr_sum[test_ind] + psnr
            '''
            print("psnr: {:.4f} ".format(psnr)) 
            x_AB_path = '{}/{}_x_AB.png'.format(test_dir, step)
            x_B_path = '{}/{}_x_B.png'.format(test_dir, step)
            x_A_path = '{}/{}_x_A.png'.format(test_dir, step)
            
            vutils.save_image(x_AB.data*0.5+0.5, x_AB_path)
            print("[*] Samples saved: {}".format(x_AB_path))
            vutils.save_image(x_B.data*0.5+0.5, x_B_path)
            print("[*] Samples saved: {}".format(x_B_path))
            vutils.save_image(x_A.data*0.5+0.5, x_A_path)
            print("[*] Samples saved: {}".format(x_A_path))
            '''

            step[test_ind] += 1
        time_end=time.time()
        print('totally testing cost',time_end-time_start)
        print(psnr_sum/step)
    def _get_variable(self, inputs):
        if self.num_gpu > 0:
            out = Variable(inputs.cuda())
        else:
            out = Variable(inputs)
        return out
