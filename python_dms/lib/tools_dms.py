import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt
from scipy.sparse import diags
import time
from dms import *
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
import torch
from torch import nn
def cross_entropy_loss2d(inputs, targets, cuda=False, balance=1.1):
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    n, c, h, w = inputs.size()
    weights = np.zeros((n, c, h, w))
    for i in range(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * balance / valid
    weights = torch.Tensor(weights)
    if cuda:
        weights = weights.cuda()
#     print(weights)
    inputs = torch.sigmoid(inputs)
    loss=nn.BCELoss(weights,reduction='mean')(inputs,targets)     #loss=nn.BCELoss(weights,size_average=False)(inputs,targets)
    return loss

def save_results(a,b,c,link):
    
    np.save(link+"a",a)
    np.save(link+"b",b)
    np.save(link+"c",c)    


def normalization(x):
    z = (x - np.min(x)) / (np.max(x) - np.min(x))
    return z
def optD(x):
    rows,cols = x.shape
    y=  np.zeros((rows,cols,2))
    y[:, :, 0] = np.concatenate((x[:, 1:] - x[:, 0:-1], np.zeros((rows, 1))),axis=1) / 2.
    y[:, :, 1] = np.concatenate((x[1:, :] - x[0:-1, :], np.zeros((1, cols))),axis=0) / 2.
    return y
def optD_normalized(x):
    rows, cols = x.shape
    y = np.zeros((rows, cols, 2))
    # # print(temp.shape)
    y[:, :, 0] = np.concatenate((x[:, 1:] - x[:, 0:-1], np.zeros((rows, 1))),axis=1) / 2.
    y[:, :, 1] = np.concatenate((x[1:, :] - x[0:-1, :], np.zeros((1, cols))),axis=0) / 2.
    y= np.abs(y)
    y= np.ones_like(y)*(y>0)
    return y

def draw_dots_multiresolution(b,a,beta_axis,lambda_axis,name='PSNR'):

    if name == 'Jaccard':
        stemp = 10
        vm=np.min(a[0])
        vM= np.max(a[-1])
    elif name == 'PSNR':
        vm= np.min(a[0])
        stemp = 1
        vM = np.max(a[-1])
    elif name == 'SSIM':
        vm= np.min(a[0])
        vM=np.max(a[-1])
    elif name == 'CE':
        vm= np.min(a)
        vM=np.max(a)
    y_label_list= []
    for item in beta_axis:
        y_label_list+= ['1e{}'.format(item)]
    x_label_list= []
    for item in lambda_axis:
        x_label_list+= ['1e{}'.format(item)]

    
    cm = plt.cm.get_cmap('RdYlBu')
    scale = 10

    plt.rcParams.update({'font.size': scale *4})
    size = len(b)
    # print(size)

    fig,ax = plt.subplots(figsize=(size*scale,size*scale))
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_label_list)))
    ax.set_yticks(np.arange(len(y_label_list)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_label_list)
    ax.set_yticklabels(y_label_list[::-1])

    plt.gca().invert_xaxis()
    #     plt.axhline(y=0, color='r', linestyle='-')
    #     plt.axvline(x=0, color='r', linestyle='-')
    ax.set_xlim(-2,5.5)
    ax.set_ylim(5.5,-1)
    plt.grid('on',linewidth=scale/10)
    plt.axis('equal')

    step=1
    size=size*1800   
    x= 2
    y= 2
    # plt.scatter(x,y,cmap=cm,marker='s',s=size,alpha=a[0][y,x]/30)
    # plt.scatter(x-1,y-1,color='r',marker='s',s=size,alpha=a[0][y-1,x-1]/30)
    # plt.scatter(x-1,y,color='r',marker='s',s=size,alpha=a[0][y,x-1]/30)
    # plt.scatter(x-1,y+1,color='r',marker='s',s=size,alpha=a[0][y+1,x-1]/30)
    # # plt.text(x-1,y+1,"{:10.2f}".format(a[0][y+1,x-1]))
    # # plt.text(x,y,"{:10.2f}".format(a[0][y,x]))
    # plt.scatter(x,y+1,color='r',marker='s',s=size,alpha=a[0][y+1,x]/30)
    # plt.scatter(x+1,y+1,color='r',marker='s',s=size,alpha=a[0][y+1,x+1]/30)
    # plt.scatter(x+1,y,color='r',marker='s',s=size,alpha=a[0][y,x+1]/30)
    # plt.scatter(x+1,y-1,color='r',marker='s',s=size,alpha=a[0][y-1,x+1]/30)
    # plt.scatter(x,y-1,color='r',marker='s',s=size,alpha=a[0][y-1,x]/30)
    
    # plt.scatter(x-2,y-2,color='r',marker='s',s=size,alpha=a[0][y-2,x-2]/30)
    # plt.scatter(x-2,y,color='r',marker='s',s=size,alpha=a[0][y,x-2]/30)
    # plt.scatter(x-2,y+2,color='r',marker='s',s=size,alpha=a[0][y+2,x-2]/30)
    # plt.scatter(x,y+2,color='r',marker='s',s=size,alpha=a[0][y+2,x]/30)
    # plt.scatter(x+2,y+2,color='r',marker='s',s=size,alpha=a[0][y+2,x+2]/30)
    # plt.scatter(x+2,y,color='r',marker='s',s=size,alpha=a[0][y,x+2]/30)
    # plt.scatter(x+2,y-2,color='r',marker='s',s=size,alpha=a[0][y-2,x+2]/30)
    # plt.scatter(x,y-2,color='r',marker='s',s=size,alpha=a[0][y-2,x]/30)
    
    # plt.scatter(x-1,y-2,color='r',marker='s',s=size,alpha=a[0][y-2,x-1]/30)
    # plt.scatter(x-1,y+2,color='r',marker='s',s=size,alpha=a[0][y+2,x-1]/30)
    # plt.scatter(x-2,y-1,color='r',marker='s',s=size,alpha=a[0][y-1,x-2]/30)
    # plt.scatter(x+2,y-1,color='r',marker='s',s=size,alpha=a[0][y-1,x+2]/30)
    # plt.scatter(x-2,y+1,color='r',marker='s',s=size,alpha=a[0][y+1,x-2]/30)
    # plt.scatter(x+2,y+1,color='r',marker='s',s=size,alpha=a[0][y+1,x+2]/30)
    # plt.scatter(x+1,y-2,color='r',marker='s',s=size,alpha=a[0][y-2,x+1]/30)
    # plt.scatter(x+1,y+2,color='r',marker='s',s=size,alpha=a[0][y+2,x+1]/30)
    tx =[]
    ty =[]
    tz =[]
    ts =[]
    for i in range(-2,3):
        for j in range(-2,3):
            tx += [x+i]
            ty += [y+j]
            tz += [a[0][y+j,x+i]]
            # ts += [stemp*300*a[0][y+j,x+i]]
            ts += [size]
            # plt.scatter(x+i,y+j,c=a[0][y+j,x+i]/30,marker='s',s=a[0][y+j,x+i]*300,alpha=a[0][y+j,x+i]/30)
            # plt.colorbar()

            plt.text(x+i-0.2,y+j,"{:10.2f}".format(a[0][y+j,x+i]),color='r')
    plt.scatter(tx,ty,c=tz,s=ts,cmap='winter',vmin=vm,vmax=vM,marker='s')
    # plt.colorbar()

#
    plt.rcParams.update({'font.size': scale*2})
    # plt.text(x,y,str(dot))
    # zoom in optimal position

    dot = b[0]
    size=size/3
    xop,yop = dot[1],dot[0]
    x = x- (2*step-xop*step)
    y = y- (2*step-yop*step)
    # plt.scatter(x,y,color='r',marker='x',s=size*3,alpha=1)
    # print(x,y)
    # print(xop,yop)


    step =  step /2
    tx =[]
    ty =[]
    tz =[]
    ts =[]
    for i in range(-2,3):
        for j in range(-2,3):
            # print(i,j)
            tx += [x+i*step]
            ty += [y+j*step]
            tz += [a[1][2+j,2+i]]
            # ts += [a[1][2+j,2+i]*100*stemp]
            ts += [size]
            # plt.scatter(x+i*step,y+j*step,color='b',marker='s',s=a[1][2+j,2+i]*100,alpha=a[1][2+j,2+i]/30)
            # plt.text(x+i*step-step/2,y+j*step,"{:10.2f}".format(a[1][2+j,2+i]),color='b')
            # plt.scatter(x+i*step,y+j*step,color='b',marker='s',s=a[1][yop+j,xop+i]*100,alpha=a[1][yop+j,xop+i]/30)
            # 
    plt.scatter(tx,ty,c=tz,s=ts,marker='s',cmap='winter',vmin=vm,vmax=vM)
    

    # plt.scatter(x,y,color='b',marker='x',s=size)
    # plt.scatter(x+step,y-2*step,color='b',marker='s',s=size)
    # plt.scatter(x-step,y-2*step,color='b',marker='s',s=size)
    
    # plt.scatter(x-2*step,y-step,color='b',marker='s',s=size)
    # plt.scatter(x-step,y-step,color='b',marker='s',s=size)
    # plt.scatter(x,y-step,color='b',marker='s',s=size)
    # plt.scatter(x+step,y-step,color='b',marker='s',s=size)
    # plt.scatter(x+2*step,y-step,color='b',marker='s',s=size)
    
    # plt.scatter(x+step,y,color='b',marker='s',s=size)
    # plt.scatter(x-step,y,color='b',marker='s',s=size)
    
    # plt.scatter(x-2*step,y+step,color='b',marker='s',s=size)
    # plt.scatter(x-step,y+step,color='b',marker='s',s=size)
    # plt.scatter(x,y+step,color='b',marker='s',s=size)
    # plt.scatter(x+step,y+step,color='b',marker='s',s=size)
    # plt.scatter(x+2*step,y+step,color='b',marker='s',s=size)
    
    # plt.scatter(x+step,y+2*step,color='b',marker='s',s=size)
    # plt.scatter(x-step,y+2*step,color='b',marker='s',s=size)

    # colors='b'

    # if dot[0]== 4:
    #     plt.scatter(x-2*step,y+2*step,c='b',marker='s',s=size)
    #     plt.scatter(x-step,y+2*step,c='b',marker='s',s=size)
    #     plt.scatter(x,y+2*step,c='b',marker='s',s=size)
    #     plt.scatter(x+step,y+2*step,c='b',marker='s',s=size)
    #     plt.scatter(x+2*step,y+2*step,c='b',marker='s',s=size)

    # if dot[1]==4:
    #     plt.scatter(x+2*step,y+2*step,c=colors,marker='s',s=size)
    #     plt.scatter(x+2*step,y-step,c=colors,marker='s',s=size)
    #     plt.scatter(x+2*step,y,c=colors,marker='s',s=size)
    #     plt.scatter(x+2*step,y+step,c=colors,marker='s',s=size)
    #     plt.scatter(x+2*step,y+2*step,c=colors,marker='s',s=size)

    # if dot[0]==0:
    #         plt.scatter(x-2*step,y-2*step,c=colors,marker='s',s=size)
    #         plt.scatter(x-step,y-2*step,c=colors,marker='s',s=size)
    #         plt.scatter(x,y-2*step,c=colors,marker='s',s=size)
    #         plt.scatter(x+step,y-2*step,c=colors,marker='s',s=size)
    #         plt.scatter(x+2*step,y-2*step,c=colors,marker='s',s=size)
            
    # if dot[1]==0:
    #     plt.scatter(x-2*step,y+2*step,c=colors,marker='s',s=size)
    #     plt.scatter(x-2*step,y-step,c=colors,marker='s',s=size)
    #     plt.scatter(x-2*step,y,c=colors,marker='s',s=size)
    #     plt.scatter(x-2*step,y+step,c=colors,marker='s',s=size)
    #     plt.scatter(x-2*step,y+2*step,c=colors,marker='s',s=size)

    ctab = "pygkmcb"

    for k in range(1,len(b)-1):
        colors = ctab[k]
        dot = b[k]
    
        xop,yop = dot[1],dot[0]

        x = x- (2*step-xop*step)
        y = y- (2*step-yop*step)
        size=size/3
        step =  step /2
        tx =[]
        ty =[]
        tz =[]
        ts =[]
        for i in range(-2,3):
            for j in range(-2,3):
                # print(i,j)
                tx += [x+i*step]
                ty += [y+j*step]
                tz += [a[k+1][2+j,2+i]]
                # ts += [a[k][2+j,2+i]*10*stemp]
                ts += [size]
        # for i in range(-2,3):
        #     for j in range(-2,3):
        #         plt.scatter(x+i*step,y+j*step,color=colors,marker='s',s=a[k][2+j,2+i]*50/(2**k),alpha=a[k][2+j,2+i]/30)
        #         plt.rcParams.update({'font.size': scale/(k)})
                # plt.text(x+i*step-step/2,y+j*step,"{:10.2f}".format(a[k+1][2+j,2+i]),color=colors)
        plt.rcParams.update({'font.size': scale/(k)})
        cax=plt.scatter(tx,ty,c=tz,s=ts,marker='s',cmap='winter',vmin=vm,vmax=vM)

        # plt.scatter(x,y,color=colors,marker='x',s=size,alpha=1)
        # plt.scatter(x+step,y-2*step,c=colors,marker='s',s=size)
        # plt.scatter(x-step,y-2*step,c=colors,marker='s',s=size)

        # plt.scatter(x-2*step,y-step,c=colors,marker='s',s=size)
        # plt.scatter(x-step,y-step,c=colors,marker='s',s=size)
        # plt.scatter(x,y-step,c=colors,marker='s',s=size)
        # plt.scatter(x+step,y-step,c=colors,marker='s',s=size)
        # plt.scatter(x+2*step,y-step,c=colors,marker='s',s=size)

        # plt.scatter(x+step,y,c=colors,marker='s',s=size)
        # plt.scatter(x-step,y,c=colors,marker='s',s=size)

        # plt.scatter(x-2*step,y+step,c=colors,marker='s',s=size)
        # plt.scatter(x-step,y+step,c=colors,marker='s',s=size)
        # plt.scatter(x,y+step,c=colors,marker='s',s=size)
        # plt.scatter(x+step,y+step,c=colors,marker='s',s=size)
        # plt.scatter(x+2*step,y+step,c=colors,marker='s',s=size)

        # plt.scatter(x+step,y+2*step,c=colors,marker='s',s=size)
        # plt.scatter(x-step,y+2*step,c=colors,marker='s',s=size)
        # tx =[]
        # ty =[]
        # tz =[]
        # ts =[]
        # if dot[0]==0:
        #     for i in range(-2,3):
        #         tx += [x+i*step]
        #         ty += [y-2*step]
        #         tz += [a[k][2+j,2+i]*50/(2**k)]
        #         ts += [a[k][2+j,2+i]]
            # plt.scatter(x-2*step,y-2*step,cmap='viridis',marker='s',s=size)
            # plt.scatter(x-step,y-2*step,cmap='viridis',marker='s',s=size)
            # plt.scatter(x,y-2*step,cmap='viridis',marker='s',s=size)
            # plt.scatter(x+step,y-2*step,cmap='viridis',marker='s',s=size)
            # plt.scatter(x+2*step,y-2*step,cmap='viridis',marker='s',s=size)
            
        # if dot[1]==0:
        #     for i in range(-2,3):
        #         tx += [x-2*step]
        #         ty += [y+i*step]
        #         tz += [a[k][2+j,2+i]*50/(2**k)]
        #         ts += [a[k][2+j,2+i]]
            # plt.scatter(x-2*step,y+2*step,cmap='viridis',marker='s',s=size)
            # plt.scatter(x-2*step,y-step,cmap='viridis',marker='s',s=size)
            # plt.scatter(x-2*step,y,cmap='viridis',marker='s',s=size)
            # plt.scatter(x-2*step,y+step,cmap='viridis',marker='s',s=size)
            # plt.scatter(x-2*step,y+2*step,cmap='viridis',marker='s',s=size)

        # if dot[0]== 4:
        #     for i in range(-2,3):
        #         tx += [x+i*step]
        #         ty += [y+2*step]
        #         tz += [a[k][2+j,2+i]*50/(2**k)]
        #         ts += [a[k][2+j,2+i]]
            # plt.scatter(x-2*step,y+2*step,cmap='viridis',marker='s',s=size)
            # plt.scatter(x-step,y+2*step,cmap='viridis',marker='s',s=size)
            # plt.scatter(x,y+2*step,cmap='viridis',marker='s',s=size)
            # plt.scatter(x+step,y+2*step,cmap='viridis',marker='s',s=size)
            # plt.scatter(x+2*step,y+2*step,cmap='viridis',marker='s',s=size)
        # if dot[1]==4:
        #     for i in range(-2,3):
        #         tx += [x+2*step]
        #         ty += [y+i*step]
        #         tz += [a[k][2+j,2+i]*50/(2**k)]
        #         ts += [a[k][2+j,2+i]]
    # dot = b[-1]
    # xop,yop = dot[1],dot[0]

    # x = x- (2*step-xop*step)
    # y = y- (2*step-yop*step)
    # size=size/3
    # step =  step /2

    # plt.scatter(x,y,color='r',marker='x',s=size)
    # plt.text(x,y,str(dot))
    
    cb=fig.colorbar(cax,ticks=[vm ,vM],orientation='vertical')   
    cb.ax.tick_params(labelsize=scale*6)
    # cb.ax.set_yticklabels([str(vm),str(vM)]) 
    plt.show()
    

def draw_multiresolution(a1,b1,color='Greens',add_text=True,vm=0,vM=1):    
    temp = np.copy(a1[0])
    size = temp.shape[0]
    row_curr = 0
    col_curr = 0
    row_prev = 0
    col_prev = 0
    a_curr= None
    a_prev= None
    r = 0
    for r in range(len(b1)):
        a_curr = np.zeros((size,size))
        print(b1[r])

        if r==0:
            a_curr = np.copy(a1[r])
            fig,ax = plt.subplots(figsize=(size,size))
            plt.imshow(a_curr,color,vmin=vm,vmax=vM)
            if add_text == True:
                for i in range(size):
                    for j in range(size):
                        text = ax.text(j, i, format(a_curr[i,j], '.2f'),
                            ha="center", va="center", color="blue")

            row_optim,col_optim = b1[r]

            row_curr = row_optim
            col_curr = col_optim

            row_prev = row_curr
            col_prev = col_curr
            print(row_prev,col_prev,row_curr,col_curr)
            plt.plot([col_prev-1.5,col_prev-1.5],[row_prev+1.5,row_prev-1.5],'g-',linewidth=5)
            plt.plot([col_prev-1.5,col_prev+1.5],[row_prev+1.5,row_prev+1.5],'g-',linewidth=5)
            plt.plot([col_prev-1.5,col_prev+1.5],[row_prev-1.5,row_prev-1.5],'g-',linewidth=5)
            plt.plot([col_prev+1.5,col_prev+1.5],[row_prev-1.5,row_prev+1.5],'g-',linewidth=5)
            plt.colorbar()
            plt.show()
            a_prev = a_curr


        else:
            fig,ax = plt.subplots(figsize=(size,size))

            row_optim,col_optim = b1[r-1]
            row_curr = row_prev*2
            col_curr = col_prev*2

            row_optim = row_curr +b1[r][0] -2
            col_optim = col_curr +b1[r][1] -2

            print(row_prev,col_prev,row_curr,col_curr)

            for i in range(size//2):
                for j in range(size//2):
                    a_curr[i*2,j*2]     = a_prev[i,j]
                    a_curr[i*2+1,j*2]   = a_prev[i,j]
                    a_curr[i*2,j*2+1]   = a_prev[i,j]
                    a_curr[i*2+1,j*2+1] = a_prev[i,j]

            a_curr[row_curr-2:row_curr+3,col_curr-2:col_curr+3]= a1[r]  
            a_prev = a_curr

    #         plt.plot([col_curr,0],[row_curr,0],'k-',linewidth=5)
            plt.plot([col_curr-2.5,col_curr+2.5],[row_curr-2.5,row_curr-2.5],'k-',linewidth=5)
            plt.plot([col_curr-2.5,col_curr+2.5],[row_curr+2.5,row_curr+2.5],'k-',linewidth=5)
            plt.plot([col_curr-2.5,col_curr-2.5],[row_curr-2.5,row_curr+2.5],'k-',linewidth=5)
            plt.plot([col_curr+2.5,col_curr+2.5],[row_curr-2.5,row_curr+2.5],'k-',linewidth=5)

    #         plt.plot([col_optim,5],[row_optim,3],'g-',linewidth=5)
            plt.plot([col_optim-1.5,col_optim+1.5],[row_optim-1.5,row_optim-1.5],'g-',linewidth=5)
            plt.plot([col_optim-1.5,col_optim+1.5],[row_optim+1.5,row_optim+1.5],'g-',linewidth=5)
            plt.plot([col_optim-1.5,col_optim-1.5],[row_optim-1.5,row_optim+1.5],'g-',linewidth=5)
            plt.plot([col_optim+1.5,col_optim+1.5],[row_optim-1.5,row_optim+1.5],'g-',linewidth=5)

            row_prev = row_curr 
            col_prev = col_curr


            if add_text == True:
                for i in range(size):
                    for j in range(size):
                        text = ax.text(j, i, format(a_curr[i,j], '.2f'),
                            ha="center", va="center", color="blue")
            plt.imshow(a_curr,color,vmin=vm,vmax=vM)
            plt.colorbar()
            plt.show()
        size += size

def golden_section_map(noised_im1,im1,contours_im1,bmax=5,bmin=-5,lmax=3,lmin=-6,scale_type='10',blur_type='Gaussian',grid_size=5,max_round=10,objective='PSNR',maxiter=500,stop_crit=1e-2,method='SL-PAM',norm_type='l1',eps=0.2,eps_AT_min=0.02,time_limit=60,A=None):
    out= None
    if scale_type  =='none':
        min_beta   = 1e-6
        max_beta   = 1e4
        min_lambda = 1e-7
        max_lambda = 1e3
        
        beta_axis   = np.linspace(min_beta,max_beta,grid_size)
        lambda_axis = np.linspace(min_lambda,max_lambda,grid_size)
    
        beta_list            = [beta_axis]
        lambda_list          = [lambda_axis]
    elif scale_type == '10':
        beta_axis   = np.linspace(bmax,bmin,grid_size) # Beta y-axis decreasing, because of python imshow
        lambda_axis = np.linspace(lmin,lmax,grid_size) # Keep lambda x-axis lambda increasing,
    
        beta_list            = [beta_axis]
        lambda_list          = [lambda_axis]
    
    print('Objective:',objective)
    temp = np.zeros((grid_size,grid_size))
    temp_ast = np.zeros((grid_size,grid_size))
    tab_PSNR_out   = np.zeros((max_round,grid_size,grid_size))
    tab_coord_max_PSNR_out   = []
    tab_PSNR_max = []
    
    tab_Jaccard_out = np.zeros((max_round,grid_size,grid_size))
    tab_coord_max_Jaccard_out  =  []
    tab_Jaccard_max =[]
    
    tab_CE_out = np.zeros((max_round,grid_size,grid_size))
    tab_coord_min_CE_out  =  []
    tab_CE_min =[]

    tab_SSIM_out   =  np.zeros((max_round,grid_size,grid_size))
    tab_coord_max_SSIM_out   = []
    tab_SSIM_max = []
    
    r=0; loop = 0
    time_start = time.time()

    left   = 0
    right  = 0 
    top    = 0
    bottom = 0
    while r < max_round:
        
        beta_axis_curr   = 10**beta_list[r]
        lambda_axis_curr = 10**lambda_list[r]
        # print(beta_axis_curr)
        # print(lambda_axis_curr)
        if objective=='PSNR':   
            for i in range(grid_size):
                for j in range(grid_size):
                    test = DMS(noised_im1, '', noise_type='Gaussian',blur_type=blur_type,
                               beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], method=method,MaximumIteration=maxiter,
                               noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, dkSLPAM=1e-4,
                               optD='OptD',eps=eps,time_limit=time_limit,A=A)

                    out = test.process()
                    if method != 'PALM-AT-descent' and method != 'SLPAM-eps-descent':
                        if out[-2]<=time_limit:
                            temp[i,j]    = PSNR(out[1],im1)
                        else:
                            temp[i,j]    = -1
                    else:
                        if out[-2]<=time_limit*np.ceil(np.log2(eps/0.02)):
                            temp[i,j]    = PSNR(out[1],im1)
                        else:
                            temp[i,j]    = -1
            #                     print('CT: ',out[-2],'seconds')  
            #                     print('CT current: ',time.time()-time_start)
            draw_table(temp,beta_axis_curr,lambda_axis_curr,color='Reds',vm=np.min(temp),vM=np.max(temp))
            tab_PSNR_out[r]    = temp
            tab_PSNR_max    += [np.max(tab_PSNR_out)]
            
            coord_max_PSNR_curr    = np.unravel_index(tab_PSNR_out[r].argmax(), tab_PSNR_out[r].shape)
            # Print out best PSNR for current round
            test = DMS(noised_im1, '', noise_type='Gaussian',blur_type=blur_type,
                               beta=beta_axis_curr[coord_max_PSNR_curr[0]], lamb=lambda_axis_curr[coord_max_PSNR_curr[1]], 
                               method=method,MaximumIteration=maxiter ,
                               noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                               dkSLPAM=1e-4,optD='OptD',eps=eps,time_limit=time_limit,A=A)

            out = test.process()
            draw_result(restored_image= out[1],contour_detection= out[0])
            print('Round: ',r,' ',PSNR(out[1],im1))
            # print('coord_max_PSNR_curr: ',coord_max_PSNR_curr,'$\beta$: ',beta_axis_curr[coord_max_PSNR_curr[0]],'$\lambda$: ',lambda_axis_curr[coord_max_PSNR_curr[1]])
            # if method != 'PALM-eps-descent' and method != 'SLPAM-eps-descent':
            #     print('PSNR: ', np.round(PSNR(out[1],im1),2),'dB, ','  CT: ',out[-2],'seconds','   Iter:',len(out[3]))
            # else:
            #     print('PSNR: ',  np.round(PSNR(out[1],im1),2),'dB, ','  CT: ',out[-2],'seconds','   Iter:',out[-3])
            
            
                
            tab_coord_max_PSNR_out       += [coord_max_PSNR_curr]           
            beta_list += [np.linspace(beta_list[r][coord_max_PSNR_curr[0]]+(beta_list[r][-2]-beta_list[r][-1]),beta_list[r][coord_max_PSNR_curr[0]]-(beta_list[r][-2]-beta_list[r][-1]),grid_size)]
            lambda_list += [np.linspace(lambda_list[r][coord_max_PSNR_curr[1]]-(lambda_list[r][-1]-lambda_list[r][-2]),lambda_list[r][coord_max_PSNR_curr[1]]+(lambda_list[r][-1]-lambda_list[r][-2]),grid_size)]
            r+= 1

        elif objective=='Jaccard':
            for i in range(grid_size):
                for j in range(grid_size):

                    test = DMS(noised_im1, '', noise_type='Gaussian',blur_type=blur_type,
                               beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], method=method,MaximumIteration=maxiter,
                               noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                               dkSLPAM=1e-4,optD='OptD',eps=eps,time_limit=time_limit,eps_AT_min=eps_AT_min,A=A)

                    # print('done',i,j)
                    out = test.process()
                    # print(out[0])
                    temp[i,j]    = jaccard(out[0],contours_im1)
                    # if out[-2]<=time_limit:
                    #         temp[i,j]    = jaccard(out[0],contours_im1)
                    # else:
                    #     temp[i,j]    = -1
                    if method != 'PALM-AT-descent' and method != 'SLPAM-eps-descent':
                        if out[-2]<=time_limit:
                            temp[i,j]    = jaccard(out[0],contours_im1)
                        else:
                            temp[i,j]    = -1
                    else:
                        if out[-2]<=time_limit*np.ceil(np.log2(eps/eps_AT_min)):
                            temp[i,j] = jaccard(out[0],contours_im1)
                        else:
                            temp[i,j]    = -1
            # draw_table(temp,beta_axis_curr,lambda_axis_curr,color='Greens',vm=0,vM=1)

            tab_Jaccard_out[r]  = temp
            tab_Jaccard_max    += [np.max(temp)]
            coord_max_Jaccard_curr = np.unravel_index(tab_Jaccard_out[r].argmax(), tab_Jaccard_out[r].shape)

            test = DMS(noised_im1, '', noise_type='Gaussian',blur_type=blur_type,
                               beta=beta_axis_curr[coord_max_Jaccard_curr[0]], lamb=lambda_axis_curr[coord_max_Jaccard_curr[1]], 
                               method=method,MaximumIteration=maxiter ,
                               noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit,
                               dkSLPAM=1e-4,optD='OptD',eps=eps,eps_AT_min=eps_AT_min,time_limit=time_limit,A=A)

            out = test.process()
            # draw_result(restored_image= out[1],contour_detection= out[0],gth_contour=contours_im1)


            print('Round: ',r, ' ',  jaccard(out[0],contours_im1))
            # print('coord_max_Jaccard_curr: ',coord_max_Jaccard_curr,r'$beta$: ',beta_axis_curr[coord_max_Jaccard_curr[0]],'$lambda$: ',lambda_axis_curr[coord_max_Jaccard_curr[1]])
            
            # if method != 'PALM-eps-descent' and method != 'SLPAM-eps-descent':
            #     print('Jaccard: ', np.round(jaccard(out[0],contours_im1),3),'PSNR:',np.round(PSNR(out[1],im1),2),'dB ,')
            # else:
            #     print('Jaccard: ', np.round(jaccard(out[0],contours_im1),3),'PSNR:',np.round(PSNR(out[1],im1),2),'dB ,','  CT: ',out[-1],'seconds','   Iter:',out[-2])
            
            tab_coord_max_Jaccard_out       += [coord_max_Jaccard_curr]
            beta_list += [np.linspace(beta_list[r][coord_max_Jaccard_curr[0]]+(beta_list[r][-2]-beta_list[r][-1]),beta_list[r][coord_max_Jaccard_curr[0]]-(beta_list[r][-2]-beta_list[r][-1]),grid_size)]
            lambda_list += [np.linspace(lambda_list[r][coord_max_Jaccard_curr[1]]-(lambda_list[r][-1]-lambda_list[r][-2]),lambda_list[r][coord_max_Jaccard_curr[1]]+(lambda_list[r][-1]-lambda_list[r][-2]),grid_size)]
            r+= 1
        elif objective=='CE':
            for i in range(grid_size):
                for j in range(grid_size):

                    test = DMS(noised_im1, '', noise_type='Gaussian',blur_type=blur_type,
                               beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], method=method,MaximumIteration=maxiter,
                               noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                               dkSLPAM=1e-4,optD='OptD',eps=eps,time_limit=time_limit,eps_AT_min=eps_AT_min,A=A)

                    out = test.process()                    
                    cont_rec_torch= np.moveaxis(out[0],-1,0)
                    e_exacte_torch= np.moveaxis(contours_im1,-1,0)
                    cont_rec_torch = torch.tensor([cont_rec_torch],dtype=torch.float)
                    e_exacte_torch = torch.tensor([e_exacte_torch],dtype=torch.float)
                    if method != 'PALM-AT-descent' and method != 'SLPAM-eps-descent':
                        if out[-2]<=time_limit:
                            temp[i,j]    = cross_entropy_loss2d(inputs=cont_rec_torch,targets=e_exacte_torch)
                        else:
                            temp[i,j]    = -1
                    else:
                        if out[-2]<=time_limit*np.ceil(np.log2(eps/eps_AT_min)):
                            temp[i,j] = cross_entropy_loss2d(inputs=cont_rec_torch,targets=e_exacte_torch)
                        else:
                            temp[i,j]    = -1
            tab_CE_out[r]  = temp
            tab_CE_min    += [np.min(temp)]
            coord_min_CE_curr = np.unravel_index(tab_CE_out[r].argmin(), tab_CE_out[r].shape)

            test = DMS(noised_im1, '', noise_type='Gaussian',blur_type=blur_type,
                               beta=beta_axis_curr[coord_min_CE_curr[0]], lamb=lambda_axis_curr[coord_min_CE_curr[1]], 
                               method=method,MaximumIteration=maxiter ,
                               noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit,
                               dkSLPAM=1e-4,optD='OptD',eps=eps,eps_AT_min=eps_AT_min,time_limit=time_limit,A=A)

            out = test.process()


            print('Round: ',r, 'iteration',len(out[3]),  cross_entropy_loss2d(inputs=cont_rec_torch,targets=e_exacte_torch))
            
            tab_coord_min_CE_out       += [coord_min_CE_curr]
            beta_list += [np.linspace(beta_list[r][coord_min_CE_curr[0]]+(beta_list[r][-2]-beta_list[r][-1]),beta_list[r][coord_min_CE_curr[0]]-(beta_list[r][-2]-beta_list[r][-1]),grid_size)]
            lambda_list += [np.linspace(lambda_list[r][coord_min_CE_curr[1]]-(lambda_list[r][-1]-lambda_list[r][-2]),lambda_list[r][coord_min_CE_curr[1]]+(lambda_list[r][-1]-lambda_list[r][-2]),grid_size)]
            r+= 1
                
        elif objective=='SSIM':
            for i in range(grid_size):
                for j in range(grid_size):
                    test = DMS(noised_im1, '', noise_type='Gaussian',blur_type=blur_type,
                               beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], method=method,MaximumIteration=maxiter,
                               noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                               dkSLPAM=1e-4,optD='OptD',eps=eps,time_limit=time_limit)

                    out = test.process()
                    if method != 'PALM-AT-descent' and method != 'SLPAM-eps-descent':
                        if out[-2]<=time_limit:
                            temp[i,j]    = ssim(out[1],im1)
                        else:
                            temp[i,j]    = -1
                    else:
                        if out[-2]<=time_limit*np.ceil(np.log2(eps/0.02)):
                            temp[i,j] = temp[i,j]    = ssim(out[1],im1)
                        else:
                            temp[i,j]    = -1

            draw_table(temp,beta_axis_curr,lambda_axis_curr,color='Purples',vm=0,vM=1)
            tab_SSIM_out[r]  = temp
            tab_SSIM_max    += [np.max(temp)]
            coord_max_SSIM_curr = np.unravel_index(tab_SSIM_out[r].argmax(), tab_SSIM_out[r].shape)
            # Print out best SSIM for current round
            test = DMS(noised_im1, '', noise_type='Gaussian',blur_type='none',
                               beta=beta_axis_curr[coord_max_SSIM_curr[0]], lamb=lambda_axis_curr[coord_max_SSIM_curr[1]], 
                               method=method,MaximumIteration=maxiter ,
                               noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                               dkSLPAM=1e-4,optD='OptD',eps=eps,time_limit=time_limit)

            out = test.process()
            draw_result(restored_image= out[1],contour_detection= out[0])
            print('Round: ',r)
            print('coord_max_SSIM_curr: ',coord_max_SSIM_curr,r'$\beta$: ',beta_axis_curr[coord_max_SSIM_curr[0]],'$\lambda$: ',lambda_axis_curr[coord_max_SSIM_curr[1]])
            
            if method != 'PALM-eps-descent' and method != 'SLPAM-eps-descent':
                print('SSIM: ', ssim(out[1],im1),'CT: ',out[-2],'seconds','   Iter:',len(out[3]))
            else:
                print('SSIM: ',  ssim(out[1],im1),'  CT: ',out[-2],'seconds','   Iter:',out[-3])
            
            # if (coord_max_SSIM_curr[0]>0 and coord_max_SSIM_curr[0]<grid_size-1) and (coord_max_SSIM_curr[1]>0 and coord_max_SSIM_curr[1]<grid_size-1):
            tab_coord_max_SSIM_out       += [coord_max_SSIM_curr]
            beta_list += [np.linspace(beta_list[r][coord_max_SSIM_curr[0]]+(beta_list[r][-2]-beta_list[r][-1]),beta_list[r][coord_max_SSIM_curr[0]]-(beta_list[r][-2]-beta_list[r][-1]),grid_size)]
            lambda_list += [np.linspace(lambda_list[r][coord_max_SSIM_curr[1]]-(lambda_list[r][-1]-lambda_list[r][-2]),lambda_list[r][coord_max_SSIM_curr[1]]+(lambda_list[r][-1]-lambda_list[r][-2]),grid_size)]
            r+= 1
                
    print('\n\n\n')
    print('Meaningful r (optimum is in the middle)', r)
    # print('Total loop',loop)
    time_elapsed = (time.time() - time_start)
    print('Total Computation time:', time_elapsed)
    if objective == 'PSNR':
        print('PSNR out')
        return tab_PSNR_out,tab_coord_max_PSNR_out,tab_PSNR_max,out[1],out[0]
    elif objective =='Jaccard':
        return tab_Jaccard_out,tab_coord_max_Jaccard_out,tab_Jaccard_max,out[1],out[0]
    elif objective =='SSIM':
        return tab_SSIM_out,tab_coord_max_SSIM_out,tab_SSIM_max,out[1],out[0]
    elif objective =='CE':
        return tab_CE_out,tab_coord_min_CE_out,tab_CE_min,out[1],out[0]

def grid_search(noised_im1,im1,contours_im1,bmax=5,bmin=-5,lmax=3,lmin=-6,scale_type='10',blur_type='none',grid_size=5,max_round=10,objective='PSNR',maxiter=500,stop_crit=1e-2,method='SL-PAM',norm_type='l1',eps=0.2,time_limit=60,A=None):  
    
    if scale_type  =='none':
        min_beta   = 1e-6
        max_beta   = 1e4
        min_lambda = 1e-7
        max_lambda = 1e3
        
        beta_axis   = np.linspace(min_beta,max_beta,grid_size)
        lambda_axis = np.linspace(min_lambda,max_lambda,grid_size)
    
        beta_list            = [beta_axis]
        lambda_list          = [lambda_axis]
    elif scale_type == '10':
        beta_axis   = np.linspace(bmax,bmin,grid_size) # Beta y-axis decreasing, because of python imshow
        lambda_axis = np.linspace(lmin,lmax,grid_size) # Keep lambda x-axis lambda increasing,
    
        beta_list            = [beta_axis]
        lambda_list          = [lambda_axis]
    
    print('Objective:',objective)
    temp = np.zeros((grid_size,grid_size))
    tab_PSNR_out   = np.zeros((max_round,grid_size,grid_size))
    tab_coord_max_PSNR_out   = []
    tab_PSNR_max = []
    
    tab_Jaccard_out = np.zeros((max_round,grid_size,grid_size))
    tab_coord_max_Jaccard_out  =  []
    tab_Jaccard_max =[]

    tab_SSIM_out   =  np.zeros((max_round,grid_size,grid_size))
    tab_coord_max_SSIM_out   = []
    tab_SSIM_max = []
    
    time_start = time.time()
    
    beta_axis_curr   = 10**beta_list[0]
    lambda_axis_curr = 10**lambda_list[0]
    
    if objective=='PSNR':
        for i in range(grid_size):
            for j in range(grid_size):
                test = DMS(noised_im1, '', noise_type='Gaussian',blur_type=blur_type,
                           beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], method=method,MaximumIteration=maxiter,
                           noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, dkSLPAM=1e-4,
                           optD='OptD',eps=eps,time_limit=time_limit,A=A)

                out = test.process()
                if out[-2]<=time_limit:
                    temp[i,j]    = PSNR(out[1],im1)
                else:
                    temp[i,j]    = -1
        #                     print('CT: ',out[-1],'seconds')  
        #                     print('CT current: ',time.time()-time_start)
        draw_table(temp,beta_axis_curr,lambda_axis_curr,color='winter',vm=0,vM=28)
        tab_PSNR_out[0]    = temp
        tab_PSNR_max    += [np.max(tab_PSNR_out)]
        
        coord_max_PSNR_curr    = np.unravel_index(tab_PSNR_out[0].argmax(), tab_PSNR_out[0].shape)
        # Print out best PSNR for current round
        tab_coord_max_PSNR_out       += [coord_max_PSNR_curr]

        test = DMS(noised_im1, '', noise_type='Gaussian',blur_type=blur_type,
                           beta=beta_axis_curr[coord_max_PSNR_curr[0]], lamb=lambda_axis_curr[coord_max_PSNR_curr[1]], 
                           method=method,MaximumIteration=maxiter ,
                           noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                           dkSLPAM=1e-4,optD='OptD',eps=eps,time_limit=time_limit,A=A)

        out = test.process()
        draw_result(restored_image= out[1],contour_detection= out[0])
        print('Round: ',0)
        print('coord_max_PSNR_curr: ',coord_max_PSNR_curr,'beta: ',beta_axis_curr[coord_max_PSNR_curr[0]],'lambda: ',lambda_axis_curr[coord_max_PSNR_curr[1]])
        if method != 'PALM-eps-descent' and method != 'SLPAM-eps-descent':
            print('PSNR: ', PSNR(out[1],im1),'  CT: ',out[-2],'seconds','   Iter:',len(out[3]))
        else:
            print('PSNR: ', PSNR(out[1],im1),'  CT: ',out[-2],'seconds','   Iter:',out[-3])
    elif objective=='Jaccard':
        for i in range(grid_size):
            for j in range(grid_size):
                test = DMS(noised_im1, '', noise_type='Gaussian',blur_type=blur_type,
                           beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], method=method,MaximumIteration=maxiter,
                           noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                           dkSLPAM=1e-4,optD='OptD',eps=eps,time_limit=time_limit,A=A)

                out = test.process()

                if out[-2]<=time_limit:
                    temp[i,j] = jaccard(out[0],contours_im1)
                else:
                    temp[i,j]    = -1
        draw_table(temp,beta_axis_curr,lambda_axis_curr,color='winter',vm=0,vM=1)
        tab_Jaccard_out[0]  = temp
        tab_Jaccard_max    += [np.max(temp)]
        coord_max_Jaccard_curr = np.unravel_index(tab_Jaccard_out[0].argmax(), tab_Jaccard_out[0].shape)
        # Print out best Jaccard for current round
        tab_coord_max_Jaccard_out       += [coord_max_Jaccard_curr]
        test = DMS(noised_im1, '', noise_type='Gaussian',blur_type=blur_type,
                           beta=beta_axis_curr[coord_max_Jaccard_curr[0]], lamb=lambda_axis_curr[coord_max_Jaccard_curr[1]], 
                           method=method,MaximumIteration=maxiter ,
                           noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit,
                           dkSLPAM=1e-4,optD='OptD',eps=eps,time_limit=time_limit,A=A)

        out = test.process()
        draw_result(restored_image= out[1],contour_detection= out[0],gth_contour=contours_im1)
        print('Round: ',0)
        print('coord_max_Jaccard_curr: ',coord_max_Jaccard_curr,r'$\beta$: ',beta_axis_curr[coord_max_Jaccard_curr[0]],'$\lambda$: ',lambda_axis_curr[coord_max_Jaccard_curr[1]])

        if method != 'PALM-eps-descent' and method != 'SLPAM-eps-descent':
            print('Jaccard: ', jaccard(out[0],contours_im1),'CT: ',out[-2],'seconds','   Iter:',len(out[3]))
        else:
            print('Jaccard: ', jaccard(out[0],contours_im1),'  CT: ',out[-2],'seconds','   Iter:',out[-3])
        

    elif objective=='SSIM':
        for i in range(grid_size):
            for j in range(grid_size):
                test = DMS(noised_im1, '', noise_type='Gaussian',blur_type=blur_type,
                           beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], method=method,MaximumIteration=maxiter,
                           noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                           dkSLPAM=1e-4,optD='OptD',eps=eps,time_limit=time_limit,A=A)

                out = test.process()

                if out[-2]<=time_limit:
                    temp[i,j] = temp[i,j]    = ssim(out[1],im1)
                else:
                    temp[i,j]    = -1

        draw_table(temp,beta_axis_curr,lambda_axis_curr,color='Purples',vm=0,vM=1)
        tab_SSIM_out[0]  = temp
        tab_SSIM_max    += [np.max(temp)]
        coord_max_SSIM_curr = np.unravel_index(tab_SSIM_out[0].argmax(), tab_SSIM_out[0].shape)
        # Print out best SSIM for current round
        tab_coord_max_SSIM_out       += [coord_max_SSIM_curr]
        test = DMS(noised_im1, '', noise_type='Gaussian',blur_type=blur_type,
                           beta=beta_axis_curr[coord_max_SSIM_curr[0]], lamb=lambda_axis_curr[coord_max_SSIM_curr[1]], 
                           method=method,MaximumIteration=maxiter ,
                           noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                           dkSLPAM=1e-4,optD='OptD',eps=eps,time_limit=time_limit,A=A)

        out = test.process()
        draw_result(restored_image= out[1],contour_detection= out[0])
        print('Round: ',0)
        print('coord_max_SSIM_curr: ',coord_max_SSIM_curr,r'$\beta$: ',beta_axis_curr[coord_max_SSIM_curr[0]],'$\lambda$: ',lambda_axis_curr[coord_max_SSIM_curr[1]])

        if method != 'PALM-eps-descent' and method != 'SLPAM-eps-descent':
            print('SSIM: ', ssim(out[1],im1),'CT: ',out[-2],'seconds','   Iter:',len(out[3]))
        else:
            print('SSIM: ',  ssim(out[1],im1),'  CT: ',out[-2],'seconds','   Iter:',out[-3])
    if objective == 'PSNR':
        print('PSNR out')
        return tab_PSNR_out,tab_coord_max_PSNR_out,tab_PSNR_max,out[1],out[0]
    elif objective =='Jaccard':
        return tab_Jaccard_out,tab_coord_max_Jaccard_out,tab_Jaccard_max,out[1],out[0]
    
    elif objective =='SSIM':
        return tab_SSIM_out,tab_coord_max_SSIM_out,tab_SSIM_max,out[1],out[0]


def draw_table(tab,beta_axis,lambda_axis,color='Reds',vm=0,vM=28):
    fig, ax= plt.subplots(figsize=(10,10))
        
    y_label_list= []
    for item in beta_axis:
        y_label_list+= ['{:.2e}'.format(item)]
    x_label_list= []
    for item in lambda_axis:
        x_label_list+= ['{:.2e}'.format(item)]

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_label_list)))
    ax.set_yticks(np.arange(len(y_label_list)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_label_list)
    ax.set_yticklabels(y_label_list)


    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    im= plt.imshow(tab,cmap=color,vmin=np.min(tab),vmax=vM)
    # im=plt.pcolormesh(tab, edgecolors='k', linewidth=2)
    ax.set_aspect('equal')

    
    for i in range(len(y_label_list)):
        for j in range(len(x_label_list)):
            text = ax.text(j, i, format(tab[i,j], '.2f'),
            ha="center", va="center", color="blue")

    plt.rcParams["font.size"] = "15"
    fig.colorbar(im,fraction=0.06, pad=0.04,ticks=[np.min(tab), np.max(tab)])
    plt.show()
    
def draw_result(restored_image,contour_detection,gth_contour=None):
    temp = plt.figure(figsize=(10,10))
    plt.imshow(restored_image,'gray')
    plt.axis('off')
    draw_contour(contour_detection,'',fig=temp)
    # print(gth_contour)
    # if gth_contour is not None:
    #     # print('draw')
    #     draw_contour(gth_contour,'',color='b',fig=temp)
    # plt.show()


def draw_table_v2(tab,cmap,name=None,with_colorbar=False,save_link=None):
    fig, ax = plt.subplots(figsize=(35,35))
    y_label_list= []
    for item in beta_range:
        y_label_list+= [str(format(item, '.1f')) if item >=1e-3 else str(format(item, '.1f'))]
    x_label_list= []
    for item in exp_lambda_range:
        x_label_list+= [str(format(item, '.6f')) if item >=1e-3 else str(format(item, '.6'))]

    row,col = np.shape(tab)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_label_list)))
    ax.set_yticks(np.arange(len(y_label_list)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_label_list[::-1])
    ax.set_yticklabels(y_label_list[::-1])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")


    
    #     im = ax.imshow(tab[:len(y_label_list),:],cmap=cmap)

    if name =='energy':
        im = ax.imshow(tab[::-1,::-1],cmap=cmap,vmin=0,vmax=15)
        for i in range(len(y_label_list)):
            for j in range(len(x_label_list)):
                if np.abs(np.floor(tab[::-1,::-1][i,j]))<1:
                    text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.3f'),
                ha="center", va="center", color="black" if tab[::-1,::-1][i,j] < np.max(tab)/1.2 else "w")
                elif np.abs(np.floor(tab[::-1,::-1][i,j]))>10 and np.abs(np.floor(tab[::-1,::-1][i,j]))<100:
                    text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.2f'),
                ha="center", va="center", color="black" if tab[::-1,::-1][i,j] < np.max(tab)/1.2 else "w")
                else:
                    text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.2f'),
                ha="center", va="center", color="black" if tab[::-1,::-1][i,j] < np.max(tab)/1.2 else "w")
        
    elif name =='PSNR':
        im = ax.imshow(tab[::-1,::-1],cmap=cmap,vmin=5,vmax=15)
        for i in range(len(y_label_list)):
            for j in range(len(x_label_list)):
                if np.abs(np.floor(tab[::-1,::-1][i,j]))<1:
                    text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.3f'),
                ha="center", va="center", color="black" if tab[::-1,::-1][i,j] < np.max(tab)/1.2 else "w")
                elif np.abs(np.floor(tab[::-1,::-1][i,j]))>10 and np.abs(np.floor(tab[::-1,::-1][i,j]))<100:
                    text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.2f'),
                ha="center", va="center", color="black" if tab[::-1,::-1][i,j] < np.max(tab)/1.2 else "w")
                else:
                    text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.2f'),
                ha="center", va="center", color="black" if tab[::-1,::-1][i,j] < np.max(tab)/1.2 else "w")
        coord_max = np.unravel_index(tab.argmax(), tab.shape)
    #         plt.plot([col-1-1/2-coord_max[1],col-1-1/2-coord_max[1]+1],[row-1-1/2-coord_max[0],row-1-1/2-coord_max[0]],linewidth=2,c='green')
        draw_circle =plt.Circle((col-1-coord_max[1], row-1-coord_max[0]), 0.5,color='b', fill=True)
    #         draw_circle = plt.Circle((0.5, 0.5), 0.3,fill=False)

        ax.set_aspect(1)
        ax.add_artist(draw_circle)
    elif name=='time':
        im = ax.imshow(tab[::-1,::-1],cmap=cmap)
        for i in range(len(y_label_list)):
            for j in range(len(x_label_list)):
                if np.abs(np.floor(tab[::-1,::-1][i,j]))<1:
                    text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.3f'),
                ha="center", va="center", color="black" if tab[::-1,::-1][i,j] < np.max(tab)/1.2 else "w")
                elif np.abs(np.floor(tab[::-1,::-1][i,j]))>10 and np.abs(np.floor(tab[::-1,::-1][i,j]))<100:
                    text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.2f'),
                ha="center", va="center", color="black" if tab[::-1,::-1][i,j] < np.max(tab)/1.2 else "w")
                else :
                    text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.2f'),
                ha="center", va="center", color="black" if tab[::-1,::-1][i,j] < np.max(tab)/1.2 else "w")
    elif name=='difference':
        im = ax.imshow(tab[::-1,::-1],cmap=cmap,vmin=-0.3,vmax=0.3)
        for i in range(len(y_label_list)):
            for j in range(len(x_label_list)):
                text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.1f'),
            ha="center", va="center", color="black" if np.abs(tab[::-1,::-1][i,j]) < 1  else "w")
    elif name =='perimeter':
        im = ax.imshow(tab[::-1,::-1],cmap=cmap,vmin=0,vmax=3)
        for i in range(len(y_label_list)):
            for j in range(len(x_label_list)):
                text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.2f'),
            ha="center", va="center", color="black" if np.abs(tab[::-1,::-1][i,j]) < 5  else "w")
    else:    
        im = ax.imshow(tab[::-1,::-1],cmap=cmap,vmin=0,vmax=1)
        for i in range(len(y_label_list)):
            for j in range(len(x_label_list)):
                text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.3f'),
            ha="center", va="center", color="black" if tab[::-1,::-1][i,j] < 0.75  else "w")
        
        coord_max = np.unravel_index(tab.argmax(), tab.shape)
    #         plt.plot([col-1-1/2-coord_max[1],col-1-1/2-coord_max[1]+1],[row-1-1/2-coord_max[0],row-1-1/2-coord_max[0]],linewidth=2,c='green')
        draw_circle =plt.Circle((col-1-coord_max[1], row-1-coord_max[0]), 0.5,color='b', fill=True)
    #         draw_circle = plt.Circle((0.5, 0.5), 0.3,fill=False)

        ax.set_aspect(1)
        ax.add_artist(draw_circle)
        
    plt.rcParams["font.size"] = "40"
    
    # Loop over data dimensions and create text annotations.
    if with_colorbar==True:        
        if name == 'Jaccard' or name=='SSIM':
            cbar = fig.colorbar(im,fraction=0.06, pad=0.04,ticks=[0, 1])
            cbar.ax.tick_params(labelsize=150)
        elif name=='difference':
            cbar = fig.colorbar(im,fraction=0.06, pad=0.04,ticks=[-0.3, 0.3])
            cbar.ax.set_yticklabels(['< -0.3', '> 0.3']) 
            cbar.ax.tick_params(labelsize=150)
        elif name=='perimeter':
            cbar = fig.colorbar(im,fraction=0.06, pad=0.04,ticks=[0, 3])
            cbar.ax.set_yticklabels(['0', '> 3']) 
            cbar.ax.tick_params(labelsize=150)
        else:
            cbar = fig.colorbar(im,fraction=0.06, pad=0.04,ticks=[1.7, 2.7])
            cbar.ax.set_yticklabels(['1.7', '2.7']) 
            cbar.ax.tick_params(labelsize=150)
    #     plt.axis('off')
    fig.tight_layout()
    if save_link != None:
        plt.savefig(save_link)
    plt.show()

def opt_H(rows,cols):
    diagonals = [(0.5 * np.ones(cols - 1)).tolist(), (-0.5 * np.ones(cols)).tolist(), [0.5]]
    block = diags(diagonals, [1, 0, -cols + 1])
    opt_H = scp.sparse.block_diag([block for _ in range(rows)])
    return opt_H

def opt_V(rows,cols):
    diagonals = [-0.5 * np.ones(rows * cols), 0.5 * np.ones((rows - 1) * cols), 0.5 * np.ones(cols)]
    opt_V = diags(diagonals, [0, cols, -(rows * cols) + cols])
    return opt_V


def optD0_create():
    D = vstack((H,V))
    return D

def optD1_create():
    D1 = hstack((V,-H))
    return D1

def L2( x):
    temp = np.sqrt(np.sum(x ** 2))
    return temp
    
def perimeter_estimation(norm_type,method,en_SLPAM,en_PALM,eps,rows,cols):
    if norm_type=='l1':
        if method=='PALM':
            return np.sum(np.abs(en_PALM[:,:,0]/rows)+np.abs(en_PALM[:,:,1]/cols))
        elif method == 'SL-PAM':
            return np.sum(np.abs(en_SLPAM[:,:,0]/rows)+np.abs(en_SLPAM[:,:,1]/cols))
    if norm_type=='AT' or norm_type=='AT-fourier':
        if method=='PALM':
            e = en_PALM
        elif method == 'SL-PAM':
            e = en_SLPAM
        optD1 = optD1_create()
        e_ravel_0 = e[:,:,0].ravel('C')/rows
        e_ravel_1 = e[:,:,1].ravel('C')/cols
        e_ravel = np.hstack((e_ravel_0,e_ravel_1))
        optD1e = optD1.dot(e_ravel)
        return eps * L2(optD1e) ** 2 + (0.25 / eps) * L2(e_ravel) ** 2
    


    # def loop_SL_PAM_gif(self):
    #     Jn_SLPAM = 1e10 * np.ones(self.MaximumIteration + 1)
    #     self.ck = np.ones(self.MaximumIteration + 1)
    #     self.error_image_SLPAM = 1e5 * np.ones(self.MaximumIteration + 1)
    #     err = 1.
    #     self.energy(self.un_SLPAM, self.en_SLPAM, self.image_degraded)
    #
    #     it = 0
    #     dk = self.dk_SLPAM
    #     Jn_SLPAM[it] = self.J_initial
    #     # Progress bar
    #     widgets = ['Processing: ', Percentage(), ' ', Bar(marker='-', left='[', right=']'),
    #                ' ', ETA(), ' ', FileTransferSpeed()]
    #     pbar = ProgressBar(widgets=widgets, maxval=self.MaximumIteration)
    #     pbar.start()
    #
    #     # Main loop
    #     start_time = time.time()
    #     fig = plt.figure()
    #     while (it < self.MaximumIteration) and (err > self.stop_criterion):
    #         ck = self.norm_ck_dk(method='SL-PAM')
    #         self.ck[it] = ck
    #         self.un_SLPAM = self.L_prox(self.un_SLPAM - (self.beta / ck) * self.S_du(self.un, self.en_SLPAM), 1 / ck,
    #                               self.image_degraded)
    #         over = self.beta * self.S_D(self.un_SLPAM) + dk / 2. * self.en_SLPAM
    #         lower = self.beta * self.S_D(self.un_SLPAM) + dk / 2.
    #         self.en_SLPAM = self.R_prox(over / lower, self.lam / (2 * lower))
    #         self.energy(self.un_SLPAM, self.en_SLPAM, self.image_degraded)
    #         Jn_SLPAM[it + 1] = self.J
    #         err = abs(Jn_SLPAM[it + 1] - Jn_SLPAM[it])
    #         self.error_image_SLPAM[it] = np.linalg.norm(self.un_SLPAM - self.image)
    #         self.time[it] = time.time() - start_time
    #         im = plt.imshow(self.un_SLPAM,cmap='gray')
    #         self.gif_SLPAM.append([im])
    #         it += 1
    #         pbar.update(it)
    #
    #     self.it_SLPAM  = it
    #     ani = animation.ArtistAnimation(fig, self.gif_SLPAM, interval=50, blit=True,
    #                                    repeat_delay=5000)
    #     writer = PillowWriter(fps=40)
    #     ani.save("out/out_gif/[reconstructed-SL-PAM][Iteration={}][norm={}][dk={}][beta={}][lambda={}][{}-{}-{}][{}-{}][{}].gif".format(self.it_SLPAM ,self.norm_type,self.dk_SLPAM,self.beta,self.lam,self.blur_type, self.blur_size, self.blur_std,
    #                                                                    self.noise_type, self.noise_std, self.save_name), writer=writer)
    #
    #     plt.close()
    #
    #
    #     pbar.finish()
    #     self.error_curve = np.log(np.abs(Jn_SLPAM[1:it] - Jn_SLPAM[:it - 1]))
    #     self.image_reconstructed = self.normalization(self.un_SLPAM)
    #     return self.en_SLPAM, self.image_reconstructed, self.image_degraded, Jn_SLPAM[:it], self.error_curve[:it], self.time[
    #                                                                                                         :it]
    #
    # def loop_PALM_gif(self):
    #
    #     self.Jn_PALM = 1e10 * np.ones(self.MaximumIteration + 1)
    #     self.ck_PALM = np.ones(self.MaximumIteration + 1)
    #     self.dk_PALM = np.ones(self.MaximumIteration + 1)
    #     self.error_image_PALM = 1e5 * np.ones(self.MaximumIteration + 1)
    #     it = 0
    #     err = 1.
    #     self.energy(self.un_PALM, self.en_PALM, self.image_degraded)
    #     self.Jn_PALM[it] = self.J_initial
    #     # Progress bar
    #     widgets = ['Processing: ', Percentage(), ' ', Bar(marker='-', left='[', right=']'),
    #                ' ', ETA(), ' ', FileTransferSpeed()]
    #     pbar = ProgressBar(widgets=widgets, maxval=self.MaximumIteration)
    #     pbar.start()
    #     # Main loop
    #     start_time = time.time()
    #     fig2 = plt.figure()
    #     while (it < self.MaximumIteration) and (err > self.stop_criterion):
    #         ck,dk = self.norm_ck_dk(method='PALM')
    #         self.ck_PALM[it] = ck
    #         self.dk_PALM[it] = dk
    #         self.un_PALM = self.L_prox(self.un_PALM - (self.beta / ck) * self.S_du(self.un_PALM, self.en_PALM), 1 / ck,
    #                                self.image_degraded)
    #         self.en_PALM = self.R_prox(self.en_PALM - (self.beta / dk) * self.S_de(self.un_PALM, self.en_PALM), self.lam / dk)
    #         self.energy(self.un_PALM, self.en_PALM, self.image_degraded)
    #         self.Jn_PALM[it + 1] = self.J
    #         err = abs(self.Jn_PALM[it + 1] - self.Jn_PALM[it])
    #         self.error_image_PALM[it] = np.linalg.norm(self.un_PALM - self.image)
    #         self.time2[it] = time.time() - start_time
    #         im = plt.imshow(self.un_PALM, cmap='gray')
    #         self.gif_PALM.append([im])
    #         it += 1
    #         pbar.update(it)
    #
    #     self.it_PALM  = it
    #     ani2 = animation.ArtistAnimation(fig2, self.gif_PALM, interval=50, blit=True,
    #                                     repeat_delay=5000)
    #     writer = PillowWriter(fps=40)
    #     ani2.save("out/out_gif/[reconstructed-PALM][Iteration={}][norm={}][dk={}][beta={}][lambda={}][{}-{}-{}][{}-{}][{}].gif".format(
    #         self.it_PALM ,self.norm_type,self.dk_SLPAM, self.beta, self.lam, self.blur_type, self.blur_size, self.blur_std,
    #         self.noise_type, self.noise_std, self.save_name), writer=writer)
    #     plt.close()
    #     pbar.finish()
    #
    #     self.error_curve_PALM = np.log(np.abs(self.Jn_PALM[1:it] - self.Jn_PALM[:it - 1]))
    #     self.image_reconstructed_PALM = self.normalization(self.un_PALM)
    #
    #     return self.en_PALM, self.image_reconstructed_PALM, self.image_degraded, self.Jn_PALM[:it], self.error_curve_PALM[
    #                                                                                     :it], self.time2[:it]
    #
    # def process_gif(self):
    #
    #     if (self.noised_image_input is None):
    #         self.Addblur_v2()
    #         self.AddNoise()
    #
    #     if (self.method == 'SL-PAM'):
    #
    #         self.initialisation_u_e_SLPAM(type_contour=self.type_contour)
    #         contour_SL_PAM, restored_image_SL_PAM, degraded_image_SL_PAM, energy_SL_PAM, error_SL_PAM, time_exc_SL_PAM = self.loop_SL_PAM_gif()
    #         return contour_SL_PAM, restored_image_SL_PAM, degraded_image_SL_PAM, energy_SL_PAM, error_SL_PAM, time_exc_SL_PAM
    #     elif (self.method == 'PALM'):
    #         self.initialisation_u_e_PALM(type_contour=self.type_contour)
    #         contour_PALM, restored_image_PALM, degraded_image_PALM, energy_PALM, error_PALM, time_exc_PALM = self.loop_PALM_gif()
    #         return contour_PALM, restored_image_PALM, degraded_image_PALM, energy_PALM, error_PALM, time_exc_PALM
    #     elif (self.method == 'compare'):
    #         self.initialisation_u_e_SLPAM(type_contour=self.type_contour)
    #         contour_SL_PAM, restored_image_SL_PAM, degraded_image_SL_PAM, energy_SL_PAM, error_SL_PAM, time_exc_SL_PAM = self.loop_SL_PAM_gif()
    #         self.initialisation_u_e_PALM(type_contour=self.type_contour)
    #         contour_PALM, restored_image_PALM, degraded_image_PALM, energy_PALM, error_PALM, time_exc_PALM = self.loop_PALM_gif()
    #         return contour_SL_PAM, restored_image_SL_PAM, degraded_image_SL_PAM, energy_SL_PAM, time_exc_SL_PAM, contour_PALM, restored_image_PALM, degraded_image_PALM, energy_PALM, error_PALM, time_exc_PALM
