_author__ = 'linjx'
import cv2
from PIL import Image
import numpy as np
def img_random_dis(input_batch, ind = None):
    #if ind is 6:
    #imres = np.random.randint(0,6)
    if ind is None:         
        imres = np.random.randint(0,6)
    else:
        imres = ind
    img_dis=np.zeros((input_batch.shape[0],128+16*imres,96+16*imres,1))
    img_ref=np.zeros((input_batch.shape[0],128+16*imres,96+16*imres,1))
    #img_dis=np.zeros((input_batch.shape[0],256+16*imres,192+16*imres,1))
    #img_ref=np.zeros((input_batch.shape[0],256+16*imres,192+16*imres,1))
    sigma=30
    for i in range(input_batch.shape[0]):
        img_rgb  = input_batch[i]
        img_rgb = 255*(img_rgb*0.5+0.5) 
        img_rgb =  np.transpose(img_rgb,(1,2,0))
        img  = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        img  = cv2.resize(img,(96+16*imres,128+16*imres),interpolation = cv2.INTER_CUBIC)     
        #img  = cv2.resize(img,(192+16*imres,256+16*imres),interpolation = cv2.INTER_CUBIC)             
        img1 = cv2.resize(img,None,fx=1.0/4.0, fy=1.0/4.0, interpolation = cv2.INTER_CUBIC)
        noise= sigma*np.random.randn(img1.shape[0],img1.shape[1])#sigma*np.random.rand(img1.shape[0],img1.shape[1])
        img1 = img1+noise                
        img1 = cv2.resize(img1,None,fx=1.0*4.0, fy=1.0*4.0, interpolation = cv2.INTER_NEAREST)                             

        img1 = np.expand_dims(img1,axis=2)

        img  = np.expand_dims(img,axis=2)
        img_dis[i]=img1
        img_ref[i]=img

    
    img_dis = np.transpose(img_dis,(0,3,1,2))
    img_ref = np.transpose(img_ref,(0,3,1,2))
    #print img_ref.shape
    return (img_dis/255-0.5)*2,(img_ref/255-0.5)*2
