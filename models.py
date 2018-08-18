import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import math

class SGENGenerator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(SGENGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
       
        ngf = 64
  
        use_bias = True

        ex1 = [nn.Conv2d(input_nc, ngf, kernel_size=5, padding=2,
                           bias=use_bias),
                 nn.LeakyReLU(0.2)]
        ex1 += [nn.Conv2d(ngf, ngf * 2, kernel_size=5,
                                stride=2, padding=2, bias=use_bias),
                      nn.LeakyReLU(0.2)]
        self.ex1 = nn.Sequential(*ex1)   
        
        self.eX1 = nn.Sequential(*[nn.Conv2d(ngf * 2, ngf * 8, kernel_size=5,
                                stride=8, padding=2, bias=use_bias),
                      nn.LeakyReLU(0.2)])
                      
        self.ex2 = nn.Sequential(*[nn.Conv2d(ngf*2, ngf * 4, kernel_size=5,
                                stride=2, padding=2, bias=use_bias),
                      nn.LeakyReLU(0.2)])
        
        self.eX2 = nn.Sequential(*[nn.Conv2d(ngf * 4, ngf * 8, kernel_size=5,
                                stride=4, padding=2, bias=use_bias),
                      nn.LeakyReLU(0.2)])
                      
        self.ex3 = nn.Sequential(*[nn.Conv2d(ngf*4, ngf * 8, kernel_size=5,
                                stride=2, padding=2, bias=use_bias),
                      nn.LeakyReLU(0.2)])
                      
        self.eX3 = nn.Sequential(*[nn.Conv2d(ngf * 8, ngf * 8, kernel_size=5,
                                stride=2, padding=2, bias=use_bias),
                      nn.LeakyReLU(0.2)])
                      
        self.SGU1_2 = SGUBlock(ngf * 8, use_bias)
        
        self.SGU2_3 = SGUBlock(ngf * 8, use_bias)
        
        dx1 = [nn.ConvTranspose2d(ngf * 8, ngf*4, kernel_size=5, 
                                stride=2, padding=2, output_padding=1, bias=use_bias),
                 nn.ReLU(True)]
                 
        dx1 += [nn.ConvTranspose2d(ngf*4, ngf * 2, kernel_size=5,
                                stride=2, padding=2, output_padding=1, bias=use_bias),
                      nn.ReLU(True)]
        self.dx1 = nn.Sequential(*dx1)  
        
        self.dx2 = nn.Sequential(*[nn.ConvTranspose2d(ngf * 8, ngf*2, kernel_size=5, 
                                stride=4, padding=2, output_padding=3, bias=use_bias),
                 nn.ReLU(True)])
                 
        self.SGU2_1 = SGUBlock(ngf * 2, use_bias)
        
        self.dx2_1 = nn.Sequential(*[nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=5, 
                                stride=2, padding=2, output_padding=1, bias=use_bias),
                 nn.ReLU(True)])
        
        self.dx3 = nn.Sequential(*[nn.ConvTranspose2d(ngf * 8, ngf, kernel_size=5, 
                                stride=8, padding=2, output_padding=7, bias=use_bias),
                 nn.ReLU(True)])
                 
        self.SGU3_2 = SGUBlock(ngf, use_bias)
        
        self.dx3_2 = nn.Sequential(*[nn.ConvTranspose2d(ngf, ngf/2, kernel_size=5, 
                                stride=2, padding=2, output_padding=1, bias=use_bias),
                 nn.ReLU(True)])
                 
        

        self.dx4 = nn.Sequential(*[nn.Conv2d(ngf/2, output_nc, kernel_size=5, padding=2),
                 nn.Tanh()])

    def forward(self, input):
        x1 = self.ex1(input)
        X1 = self.eX1(x1)
        
        x2 = self.ex2(x1)
        X2 = self.eX2(x2)
        
        X1_2 = self.SGU1_2(X2, X1) 

        x3 = self.ex3(x2)
        X3 = self.eX3(x3)
        
        X2_3 = self.SGU2_3(X3, X1_2) 
        
        Y1 = self.dx1(X2_3)
        
        y2 = self.dx2(X1_2)
        
        Y2 = self.SGU2_1(y2, Y1)
        Y2_1 = self.dx2_1(Y2)
        
        y3 = self.dx3(X1)
        Y3 = self.SGU3_2(y3, Y2_1)
        
        Y3_2 = self.dx3_2(Y3)
        
        output = self.dx4(Y3_2)
        
        
        return output

# Define a SGU block
class SGUBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(SGUBlock, self).__init__()
        
        self.conv_block_a = nn.Sequential(*[nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias),
                       nn.LeakyReLU(0.2)])
        self.conv_block_p = nn.Sequential(*[nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias),
                       nn.LeakyReLU(0.2)])
    def forward(self, x_a, x_p):
        out = x_a * self.conv_block_a(x_a) + x_p * self.conv_block_p(x_a)
        return out

class CommonDiscriminator(nn.Module):
    def __init__(self, input_nc):
        super(CommonDiscriminator, self).__init__()
        self.input_nc = input_nc
        ndf = 64
        norm_layer = nn.BatchNorm2d
        use_bias = False
        #self.output_nc = output_nc
        #self.ngf = ngf
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ndf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ndf),
                 nn.ReLU(True)]

        n_downsampling = 3
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ndf * mult * 2),
                      nn.ReLU(True)] 
                      
        
        model_2 = [ ]
        model_2 += [nn.Linear(ndf * mult * 2, 1)]
        model_2 += [nn.Sigmoid()]                      
      
        self.model = nn.Sequential(*model)
        self.model_2 = nn.Sequential(*model_2)
        #self.net = nn.Sequential(*self.net)

    def forward(self, input):
        out = self.model(input)
        out =  torch.mean(torch.mean(out,3,keepdim=False),2, keepdim=False)#out.view(out_2.size(0), 64*8, 16, 16)
        return self.model_2(out)


