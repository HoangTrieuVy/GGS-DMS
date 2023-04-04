import numpy as np
from scipy import fftpack
import time
# from dms import *
import numpy as np
from scipy import fftpack
import time
from scipy.sparse import diags
import scipy as scp
from scipy.sparse import hstack
from scipy.sparse import vstack
from scipy.sparse import csc_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve,lgmres
import matplotlib.pyplot as plt
# import cupy as cp
# from cupyx.scipy.sparse.linalg import spsolve as spsolve_cuda
# from cupyx.scipy.sparse import csc_matrix as csc_matrix_cuda

def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img

def psf2otf(psf, output_shape=None):
    """
    Convert a point spread function to the optical transfer function.

    Parameters
    ----------
    psf : ndarray
        The point spread function of the imaging system.
    output_shape : tuple, optional
        The shape of the output OTF array. If not provided, the shape of the
        OTF will be the same as the shape of the input PSF.

    Returns
    -------
    otf : ndarray
        The optical transfer function of the imaging system.
    """
    if output_shape is None:
        output_shape = psf.shape

    # Pad the PSF to the desired output shape
    padded_psf = np.zeros(output_shape, dtype=np.complex64)
    padded_psf[:psf.shape[0], :psf.shape[1]] = psf

    # Perform the FFT on the padded PSF
    otf = np.fft.fftn(padded_psf)

    # Normalize the OTF
    otf /= np.abs(otf).max()

    return otf

def PowerIteration(A):
    bk = np.random.random(np.shape(A)[0])
    N = 25
    lambda0 = 0
    for i in range(N):
        product = A.dot(bk)
        bk = product / np.linalg.norm(product, 2)
        lambda0 = bk.dot(A.dot(bk.T)) / (bk.dot(bk.T))
    return lambda0

def draw_contour(e, name, fig=None, color="r", threshold=0.5):
    r, c, channel = np.shape(e)
    yv, xv = np.where(e[:, :, 1] > threshold)
    yh, xh = np.where(e[:, :, 0] > threshold)
    if fig == None:
        fig = plt.figure()  # ,figsize=(c//8, r//8)

    for i in range(0, len(xv)):
        plt.plot(
            [xv[i] - 0.5, xv[i] + 0.5], [(yv[i] + 0.5), (yv[i] + 0.5)], color + "-"
        )
    for i in range(0, len(xh)):
        plt.plot(
            [xh[i] + 0.5, xh[i] + 0.5], [(yh[i] - 0.5), (yh[i] + 0.5)], color + "-"
        )

def PSNR(I, Iref):
    temp = I.ravel()
    tempref = Iref.ravel()
    NbP = I.size
    EQM = np.sum((temp - tempref) ** 2) / NbP
    b = np.max(np.abs(tempref)) ** 2
    return 10 * np.log10(b / EQM)

def jaccard(im1, im2):
    """
    Computes the Jaccard metric, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    jaccard : float
        Jaccard metric returned is a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `jaccard` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)

    union = np.logical_or(im1, im2)

    return intersection.sum() / float(union.sum())

def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    # h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

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
    fig,ax = plt.subplots(figsize=(size*scale,size*scale))
    ax.set_xticks(np.arange(len(x_label_list)))
    ax.set_yticks(np.arange(len(y_label_list)))
    ax.set_xticklabels(x_label_list)
    ax.set_yticklabels(y_label_list[::-1])

    plt.gca().invert_xaxis()
    ax.set_xlim(-2,5.5)
    ax.set_ylim(5.5,-1)
    plt.grid('on',linewidth=scale/10)
    plt.axis('equal')

    step=1
    size=size*1800   
    x= 2
    y= 2
    
    tx =[]
    ty =[]
    tz =[]
    ts =[]
    for i in range(-2,3):
        for j in range(-2,3):
            tx += [x+i]
            ty += [y+j]
            tz += [a[0][y+j,x+i]]
            ts += [size]
            plt.text(x+i-0.2,y+j,"{:10.2f}".format(a[0][y+j,x+i]),color='r')
    plt.scatter(tx,ty,c=tz,s=ts,cmap='winter',vmin=vm,vmax=vM,marker='s')
    plt.rcParams.update({'font.size': scale*2})
    dot = b[0]
    size=size/3
    xop,yop = dot[1],dot[0]
    x = x- (2*step-xop*step)
    y = y- (2*step-yop*step)
    step =  step /2
    tx =[]
    ty =[]
    tz =[]
    ts =[]
    for i in range(-2,3):
        for j in range(-2,3):
            tx += [x+i*step]
            ty += [y+j*step]
            tz += [a[1][2+j,2+i]]
            ts += [size]

    plt.scatter(tx,ty,c=tz,s=ts,marker='s',cmap='winter',vmin=vm,vmax=vM)
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
                tx += [x+i*step]
                ty += [y+j*step]
                tz += [a[k+1][2+j,2+i]]
                ts += [size]
        plt.rcParams.update({'font.size': scale/(k)})
        cax=plt.scatter(tx,ty,c=tz,s=ts,marker='s',cmap='winter',vmin=vm,vmax=vM)
    cb=fig.colorbar(cax,ticks=[vm ,vM],orientation='vertical')   
    cb.ax.tick_params(labelsize=scale*6)
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

def golden_section_map(noised_im1,im1,contours_im1,bmax=5,bmin=-5,lmax=3,lmin=-6,scale_type='10',
                       grid_size=5,max_round=10,objective='Jaccard',maxiter=300,stop_crit=1e-4,method='SLPAM',norm_type='l1',eps=2.,eps_AT_min=0.02,A=None):
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
    
    r=0
    time_start = time.time()


    while r < max_round:
        
        beta_axis_curr   = 10**beta_list[r]
        lambda_axis_curr = 10**lambda_list[r]
        if objective=='PSNR':   
            for i in range(grid_size):
                for j in range(grid_size):
                    
                    test = DMS(beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], method=method,MaximumIteration=maxiter,
                               noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, dk_SLPAM_factor=1e-4,
                               optD='OptD',eps=eps,A=A)

                    out = test.process()
                    temp[i,j]   = PSNR(out[1],im1)
                        
            # draw_table(temp,beta_axis_curr,lambda_axis_curr,color='Reds',vm=np.min(temp),vM=np.max(temp))
            tab_PSNR_out[r]    = temp
            tab_PSNR_max    += [np.max(tab_PSNR_out)]
            coord_max_PSNR_curr    = np.unravel_index(tab_PSNR_out[r].argmax(), tab_PSNR_out[r].shape)
            
            # Print out best PSNR for current round
            test = DMS(beta=beta_axis_curr[coord_max_PSNR_curr[0]], lamb=lambda_axis_curr[coord_max_PSNR_curr[1]], 
                        method=method,MaximumIteration=maxiter ,
                        noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                        dk_SLPAM_factor=1e-4,optD='OptD',eps=eps,A=A)

            out = test.process()
            # draw_result(restored_image= out[1],contour_detection= out[0])
            print('Round: ',r,' ',PSNR(out[1],im1))
            tab_coord_max_PSNR_out       += [coord_max_PSNR_curr]           
            beta_list += [np.linspace(beta_list[r][coord_max_PSNR_curr[0]]+(beta_list[r][-2]-beta_list[r][-1]),beta_list[r][coord_max_PSNR_curr[0]]-(beta_list[r][-2]-beta_list[r][-1]),grid_size)]
            lambda_list += [np.linspace(lambda_list[r][coord_max_PSNR_curr[1]]-(lambda_list[r][-1]-lambda_list[r][-2]),lambda_list[r][coord_max_PSNR_curr[1]]+(lambda_list[r][-1]-lambda_list[r][-2]),grid_size)]
            r+= 1

        elif objective=='Jaccard':
            for i in range(grid_size):
                for j in range(grid_size):
                    test = DMS(#beta=2.0548170999431815,lamb=0.002058421877614818,
                                beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], 
                                method=method,MaximumIteration=maxiter,
                               noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                               dk_SLPAM_factor=1e-4,optD='OptD',eps=eps,eps_AT_min=eps_AT_min,A=A)

                    out = test.process()
                    cont_thres = np.ones_like(out[0])*(out[0]>0.5)
                    temp[i,j]    = jaccard(cont_thres,contours_im1)

            tab_Jaccard_out[r]  = temp
            tab_Jaccard_max    += [np.max(temp)]
            coord_max_Jaccard_curr = np.unravel_index(tab_Jaccard_out[r].argmax(), tab_Jaccard_out[r].shape)

            # Print out best jaccard for current round
            test = DMS( beta=beta_axis_curr[coord_max_Jaccard_curr[0]], lamb=lambda_axis_curr[coord_max_Jaccard_curr[1]], 
                        method=method,MaximumIteration=maxiter ,
                        noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit,
                        dk_SLPAM_factor=1e-4,optD='OptD',eps=eps,eps_AT_min=eps_AT_min,A=A)

            out = test.process()
            cont_thres = np.ones_like(out[0])*(out[0]>0.5)

            print('Round: ',r, ' ',  jaccard(cont_thres,contours_im1),'beta:  ',beta_axis_curr[coord_max_Jaccard_curr[0]],', lam:  ',lambda_axis_curr[coord_max_Jaccard_curr[1]] )
            tab_coord_max_Jaccard_out       += [coord_max_Jaccard_curr]
            beta_list += [np.linspace(beta_list[r][coord_max_Jaccard_curr[0]]+(beta_list[r][-2]-beta_list[r][-1]),beta_list[r][coord_max_Jaccard_curr[0]]-(beta_list[r][-2]-beta_list[r][-1]),grid_size)]
            lambda_list += [np.linspace(lambda_list[r][coord_max_Jaccard_curr[1]]-(lambda_list[r][-1]-lambda_list[r][-2]),lambda_list[r][coord_max_Jaccard_curr[1]]+(lambda_list[r][-1]-lambda_list[r][-2]),grid_size)]
            r+= 1
        elif objective=='CE':
            for i in range(grid_size):
                for j in range(grid_size):

                    test = DMS(beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], method=method,MaximumIteration=maxiter,
                               noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                               dk_SLPAM_factor=1e-4,optD='OptD',eps=eps,eps_AT_min=eps_AT_min,A=A)

                    out = test.process()                    
                    cont_rec_torch= np.moveaxis(out[0],-1,0)
                    e_exacte_torch= np.moveaxis(contours_im1,-1,0)
                    cont_rec_torch = torch.tensor([cont_rec_torch],dtype=torch.float)
                    e_exacte_torch = torch.tensor([e_exacte_torch],dtype=torch.float)
                    temp[i,j]    = cross_entropy_loss2d(inputs=cont_rec_torch,targets=e_exacte_torch)
            tab_CE_out[r]  = temp
            tab_CE_min    += [np.min(temp)]
            coord_min_CE_curr = np.unravel_index(tab_CE_out[r].argmin(), tab_CE_out[r].shape)

            test = DMS( beta=beta_axis_curr[coord_min_CE_curr[0]], lamb=lambda_axis_curr[coord_min_CE_curr[1]], 
                        method=method,MaximumIteration=maxiter ,
                        noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit,
                        dk_SLPAM_factor=1e-4,optD='OptD',eps=eps,eps_AT_min=eps_AT_min,A=A)

            out = test.process()


            print('Round: ',r, 'iteration',len(out[3]),  cross_entropy_loss2d(inputs=cont_rec_torch,targets=e_exacte_torch))
            
            tab_coord_min_CE_out       += [coord_min_CE_curr]
            beta_list += [np.linspace(beta_list[r][coord_min_CE_curr[0]]+(beta_list[r][-2]-beta_list[r][-1]),beta_list[r][coord_min_CE_curr[0]]-(beta_list[r][-2]-beta_list[r][-1]),grid_size)]
            lambda_list += [np.linspace(lambda_list[r][coord_min_CE_curr[1]]-(lambda_list[r][-1]-lambda_list[r][-2]),lambda_list[r][coord_min_CE_curr[1]]+(lambda_list[r][-1]-lambda_list[r][-2]),grid_size)]
            r+= 1
                
        elif objective=='SSIM':
            for i in range(grid_size):
                for j in range(grid_size):
                    test = DMS(beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], method=method,MaximumIteration=maxiter,
                               noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                               dkSLPAM=1e-4,optD='OptD',eps=eps)

                    out = test.process()
                    temp[i,j]    = ssim(out[1],im1)
                       

            # draw_table(temp,beta_axis_curr,lambda_axis_curr,color='Purples',vm=0,vM=1)
            tab_SSIM_out[r]  = temp
            tab_SSIM_max    += [np.max(temp)]
            coord_max_SSIM_curr = np.unravel_index(tab_SSIM_out[r].argmax(), tab_SSIM_out[r].shape)
            # Print out best SSIM for current round
            test = DMS(noised_im1, '', noise_type='Gaussian',blur_type='none',
                               beta=beta_axis_curr[coord_max_SSIM_curr[0]], lamb=lambda_axis_curr[coord_max_SSIM_curr[1]], 
                               method=method,MaximumIteration=maxiter ,
                               noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                               dkSLPAM=1e-4,optD='OptD',eps=eps)

            out = test.process()
            print('Round: ',r)
            print('coord_max_SSIM_curr: ',coord_max_SSIM_curr,r'$\beta$: ',beta_axis_curr[coord_max_SSIM_curr[0]],'$\lambda$: ',lambda_axis_curr[coord_max_SSIM_curr[1]])
            
            if method != 'PALM-eps-descent' and method != 'SLPAM-eps-descent':
                print('SSIM: ', ssim(out[1],im1),'CT: ',out[-2],'seconds','   Iter:',len(out[3]))
            else:
                print('SSIM: ',  ssim(out[1],im1),'  CT: ',out[-2],'seconds','   Iter:',out[-3])
            
            tab_coord_max_SSIM_out       += [coord_max_SSIM_curr]
            beta_list += [np.linspace(beta_list[r][coord_max_SSIM_curr[0]]+(beta_list[r][-2]-beta_list[r][-1]),beta_list[r][coord_max_SSIM_curr[0]]-(beta_list[r][-2]-beta_list[r][-1]),grid_size)]
            lambda_list += [np.linspace(lambda_list[r][coord_max_SSIM_curr[1]]-(lambda_list[r][-1]-lambda_list[r][-2]),lambda_list[r][coord_max_SSIM_curr[1]]+(lambda_list[r][-1]-lambda_list[r][-2]),grid_size)]
            r+= 1
                
    print('\n\n\n')
    print('Meaningful r (optimum is in the middle)', r)
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

def grid_search(noised_im1,im1,contours_im1,bmax=5,bmin=-5,lmax=3,lmin=-6,scale_type='10',
                grid_size=5,max_round=10,objective='PSNR',maxiter=500,stop_crit=1e-4,method='SLPAM',
                norm_type='l1',eps=0.2,A=None):  
    
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
    
    
    beta_axis_curr   = 10**beta_list[0]
    lambda_axis_curr = 10**lambda_list[0]
    
    if objective=='PSNR':
        for i in range(grid_size):
            for j in range(grid_size):
                test = DMS(beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], method=method,MaximumIteration=maxiter,
                           noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, dkSLPAM=1e-4,
                           optD='OptD',eps=eps,A=A)

                out = test.process()
                temp[i,j]    = PSNR(out[1],im1)
                
        draw_table(temp,beta_axis_curr,lambda_axis_curr,color='winter',vm=0,vM=28)
        tab_PSNR_out[0]    = temp
        tab_PSNR_max    += [np.max(tab_PSNR_out)]
        
        coord_max_PSNR_curr    = np.unravel_index(tab_PSNR_out[0].argmax(), tab_PSNR_out[0].shape)
        # Print out best PSNR for current round
        tab_coord_max_PSNR_out       += [coord_max_PSNR_curr]

        test = DMS(beta=beta_axis_curr[coord_max_PSNR_curr[0]], lamb=lambda_axis_curr[coord_max_PSNR_curr[1]], 
                method=method,MaximumIteration=maxiter ,
                noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                dkSLPAM=1e-4,optD='OptD',eps=eps,A=A)

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
                test = DMS(beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], method=method,MaximumIteration=maxiter,
                           noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                           dkSLPAM=1e-4,optD='OptD',eps=eps,A=A)

                out = test.process()

                temp[i,j] = jaccard(out[0],contours_im1)
        draw_table(temp,beta_axis_curr,lambda_axis_curr,color='winter',vm=0,vM=1)
        tab_Jaccard_out[0]  = temp
        tab_Jaccard_max    += [np.max(temp)]
        coord_max_Jaccard_curr = np.unravel_index(tab_Jaccard_out[0].argmax(), tab_Jaccard_out[0].shape)
        # Print out best Jaccard for current round
        tab_coord_max_Jaccard_out       += [coord_max_Jaccard_curr]
        test = DMS(beta=beta_axis_curr[coord_max_Jaccard_curr[0]], lamb=lambda_axis_curr[coord_max_Jaccard_curr[1]], 
                    method=method,MaximumIteration=maxiter ,
                    noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit,
                    dkSLPAM=1e-4,optD='OptD',eps=eps,A=A)

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
                test = DMS(beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], method=method,MaximumIteration=maxiter,
                           noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                           dkSLPAM=1e-4,optD='OptD',eps=eps,A=A)

                out = test.process()

                temp[i,j]    = ssim(out[1],im1)

        draw_table(temp,beta_axis_curr,lambda_axis_curr,color='Purples',vm=0,vM=1)
        tab_SSIM_out[0]  = temp
        tab_SSIM_max    += [np.max(temp)]
        coord_max_SSIM_curr = np.unravel_index(tab_SSIM_out[0].argmax(), tab_SSIM_out[0].shape)
        # Print out best SSIM for current round
        tab_coord_max_SSIM_out       += [coord_max_SSIM_curr]
        test = DMS(beta=beta_axis_curr[coord_max_SSIM_curr[0]], lamb=lambda_axis_curr[coord_max_SSIM_curr[1]], 
                method=method,MaximumIteration=maxiter ,
                noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                dkSLPAM=1e-4,optD='OptD',eps=eps,A=A)

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

