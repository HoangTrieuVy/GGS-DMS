import sys
sys.path.insert(0, '../../python_dms/lib/')

# from dms import *
from tools_dms import *
from tools_trof import *

from PIL import Image
import scipy.io as sio
import argparse
import scipy as scp
import pylab as pyl
import pywt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt
import scipy.io


## Load mires images

im1 = np.array(Image.open('original_images/dots_52_v3.png'))/255.
rows,cols = np.shape(im1)
e_exact=  np.ones((rows,cols,2))*(np.abs(optD((im1-1<0)*1))>0)
tab_std = [0.02,0.08,0.1,0.2]  # standard-deviation of noise used
simul_number = 10              # number of simulations in each std
name = 'dots_52_v3'
niter= 300
size_blur=3
std_blur= 2
out_TROF= np.zeros((simul_number,len(tab_std)))
out_PALM_l1= np.zeros((simul_number,len(tab_std)))
out_PALM_AT_eps_descent= np.zeros((simul_number,len(tab_std)))
out_PALM_AT_eps_fixed= np.zeros((simul_number,len(tab_std)))
out_Hohm= np.zeros((simul_number,len(tab_std)))
out_TROF_PSNR= np.zeros((simul_number,len(tab_std)))
out_PALM_l1_PSNR= np.zeros((simul_number,len(tab_std)))
out_PALM_AT_eps_descent_PSNR= np.zeros((simul_number,len(tab_std)))
out_PALM_AT_eps_fixed_PSNR= np.zeros((simul_number,len(tab_std)))
out_Hohm_PSNR= np.zeros((simul_number,len(tab_std)))

# out_TROF_SSIM= np.zeros((simul_number,len(tab_std)))
# out_PALM_l1_SSIM np.zeros((simul_number,len(tab_std)))
# out_PALM_AT_eps_descent_SSIM= np.zeros((simul_number,len(tab_std)))
# out_PALM_AT_eps_fixed_SSIM= np.zeros((simul_number,len(tab_std)))
# out_Hohm_SSIM= np.zeros((simul_number,len(tab_std)))


for k in range(len(tab_std)):
    for i in range(simul_number):
        noised_im1= scipy.io.loadmat('degraded_images/'+name+'_'+'noise'+'_'+str(tab_std[k])+'_'+'blur'+'_'+str(size_blur)+'_'+str(std_blur)+'_'+str(i+1)+'.mat')
        im_rec_TROF= scipy.io.loadmat('results/TROF_'+str(niter)+'_'+name+'_'+'noise'+'_'+str(tab_std[k])+'_'+'blur'+'_'+str(size_blur)+'_'+str(std_blur)+'_'+str(i+1)+'.mat')['u_rec']
        cont_rect_TROF= scipy.io.loadmat('results/TROF_'+str(niter)+'_'+name+'_'+'noise'+'_'+str(tab_std[k])+'_'+'blur'+'_'+str(size_blur)+'_'+str(std_blur)+'_'+str(i+1)+'.mat')['e_rec']

        im_rec_PALM_l1= scipy.io.loadmat('results/PALM_l1_'+str(niter)+'_'+name+'_'+'noise'+'_'+str(tab_std[k])+'_'+'blur'+'_'+str(size_blur)+'_'+str(std_blur)+'_'+str(i+1)+'.mat')['u_rec']
        cont_rect_PALM_l1= scipy.io.loadmat('results/PALM_l1_'+str(niter)+'_'+name+'_'+'noise'+'_'+str(tab_std[k])+'_'+'blur'+'_'+str(size_blur)+'_'+str(std_blur)+'_'+str(i+1)+'.mat')['e_rec']


        im_rec_PALM_AT_fixed= scipy.io.loadmat('results/PALM_AT-fourier_'+str(niter)+'_'+name+'_'+'noise'+'_'+str(tab_std[k])+'_'+'blur'+'_'+str(size_blur)+'_'+str(std_blur)+'_'+str(i+1)+'.mat')['u_rec']
        cont_rect_PALM_AT_fixed= scipy.io.loadmat('results/PALM_AT-fourier_'+str(niter)+'_'+name+'_'+'noise'+'_'+str(tab_std[k])+'_'+'blur'+'_'+str(size_blur)+'_'+str(std_blur)+'_'+str(i+1)+'.mat')['e_rec']

        im_rec_PALM_AT_descent= scipy.io.loadmat('results/PALM-eps-descent_AT-fourier_'+str(niter)+'_'+name+'_'+'noise'+'_'+str(tab_std[k])+'_'+'blur'+'_'+str(size_blur)+'_'+str(std_blur)+'_'+str(i+1)+'.mat')['u_rec']
        cont_rect_PALM_AT_descent= scipy.io.loadmat('results/PALM-eps-descent_AT-fourier_'+str(niter)+'_'+name+'_'+'noise'+'_'+str(tab_std[k])+'_'+'blur'+'_'+str(size_blur)+'_'+str(std_blur)+'_'+str(i+1)+'.mat')['e_rec']
        f=plt.figure()
        plt.axis('off')
        plt.imshow(im_rec_PALM_AT_descent,'gray',vmin=0,vmax=1)
        draw_contour(cont_rect_PALM_AT_descent,'',fig=f)
        plt.savefig('rec_'+name+'_'+'noise'+'_'+str(tab_std[k])+'_'+'blur'+'_'+str(size_blur)+'_'+str(std_blur)+'_'+str(i+1)+'.png', bbox_inches='tight', pad_inches=0)
        # plt.show()
        # f=plt.figure()
        # plt.axis('off')
        # plt.imshow(noised_im1['fNoisy'],'gray')
        # plt.savefig('noised_'+name+'_'+'noise'+'_'+str(tab_std[k])+'_'+'blur'+'_'+str(size_blur)+'_'+str(std_blur)+'_'+str(i+1)+'.png', bbox_inches='tight', pad_inches=0)

        im_rec_Hohm= scipy.io.loadmat('results/Hohm_'+name+'_'+'noise'+'_'+str(tab_std[k])+'_'+'blur'+'_'+str(size_blur)+'_'+str(std_blur)+'_'+str(i+1)+'.mat')['u_rec']
        cont_rect_Hohm= scipy.io.loadmat('results/Hohm_'+name+'_'+'noise'+'_'+str(tab_std[k])+'_'+'blur'+'_'+str(size_blur)+'_'+str(std_blur)+'_'+str(i+1)+'.mat')['e_rec']



        out_TROF[i,k] = jaccard(e_exact,cont_rect_TROF)
        out_PALM_l1[i,k] = jaccard(e_exact,cont_rect_PALM_l1)
        out_PALM_AT_eps_descent[i,k] = jaccard(e_exact,cont_rect_PALM_AT_descent)
        out_PALM_AT_eps_fixed[i,k] = jaccard(e_exact,cont_rect_PALM_AT_fixed)
        out_Hohm[i,k] = jaccard(e_exact,cont_rect_Hohm)

        out_TROF_PSNR[i,k] = PSNR(im_rec_TROF,im1)
        out_PALM_l1_PSNR[i,k] = PSNR(im_rec_PALM_l1,im1)
        out_PALM_AT_eps_descent_PSNR[i,k] = PSNR(im_rec_PALM_AT_descent,im1)
        out_PALM_AT_eps_fixed_PSNR[i,k] = PSNR(im_rec_PALM_AT_fixed,im1)
        out_Hohm_PSNR[i,k] = PSNR(im_rec_Hohm,im1)


        
# np.savetxt('out_TROF_blur.txt',out_TROF)
# np.savetxt('out_PALM_l1_blur.txt',out_PALM_l1)
# np.savetxt('out_PALM_AT_eps_descent_blur.txt',out_PALM_AT_eps_descent)
# np.savetxt('out_PALM_AT_eps_fixed_blur.txt',out_PALM_AT_eps_fixed)
# np.savetxt('out_Hohm_blur.txt',out_Hohm)

with open('out-TROF-v2-jaccard.txt', 'w') as f:
    f.write('sig max mean min \n')
    for k in range(len(tab_std)):
        f.write(str(tab_std[k])+' '+str(np.max(out_TROF[:,k]))+' '+str(np.mean(out_TROF[:,k]))+' '+str(np.min(out_TROF[:,k]))+'\n')

with open('out-Hohm-v2-jaccard.txt', 'w') as f:
    f.write('sig max mean min \n')
    for k in range(len(tab_std)):
        f.write(str(tab_std[k])+' '+str(np.max(out_Hohm[:,k]))+' '+str(np.mean(out_Hohm[:,k]))+' '+str(np.min(out_Hohm[:,k]))+'\n')

with open('out-PALM-AT-eps-descent-v2-jaccard.txt', 'w') as f:
    f.write('sig max mean min \n')
    for k in range(len(tab_std)):
        f.write(str(tab_std[k])+' '+str(np.max(out_PALM_AT_eps_descent[:,k]))+' '+str(np.mean(out_PALM_AT_eps_descent[:,k]))+' '+str(np.min(out_PALM_AT_eps_descent[:,k]))+'\n')

with open('out-PALM-AT-eps-fixed-v2-jaccard.txt', 'w') as f:
    f.write('sig max mean min \n')
    for k in range(len(tab_std)):
        f.write(str(tab_std[k])+' '+str(np.max(out_PALM_AT_eps_fixed[:,k]))+' '+str(np.mean(out_PALM_AT_eps_fixed[:,k]))+' '+str(np.min(out_PALM_AT_eps_fixed[:,k]))+'\n')

with open('out-PALM-l1-v2-jaccard.txt', 'w') as f:
    f.write('sig max mean min \n')
    for k in range(len(tab_std)):
        f.write(str(tab_std[k])+' '+str(np.max(out_PALM_l1[:,k]))+' '+str(np.mean(out_PALM_l1[:,k]))+' '+str(np.min(out_PALM_l1[:,k]))+'\n')



with open('out-TROF-v2-PSNR.txt', 'w') as f:
    f.write('sig max mean min \n')
    for k in range(len(tab_std)):
        f.write(str(tab_std[k])+' '+str(np.max(out_TROF_PSNR[:,k]))+' '+str(np.mean(out_TROF_PSNR[:,k]))+' '+str(np.min(out_TROF_PSNR[:,k]))+'\n')

with open('out-Hohm-v2-PSNR.txt', 'w') as f:
    f.write('sig max mean min \n')
    for k in range(len(tab_std)):
        f.write(str(tab_std[k])+' '+str(np.max(out_Hohm_PSNR[:,k]))+' '+str(np.mean(out_Hohm_PSNR[:,k]))+' '+str(np.min(out_Hohm_PSNR[:,k]))+'\n')

with open('out-PALM-AT-eps-descent-v2-PSNR.txt', 'w') as f:
    f.write('sig max mean min \n')
    for k in range(len(tab_std)):
        f.write(str(tab_std[k])+' '+str(np.max(out_PALM_AT_eps_descent_PSNR[:,k]))+' '+str(np.mean(out_PALM_AT_eps_descent_PSNR[:,k]))+' '+str(np.min(out_PALM_AT_eps_descent_PSNR[:,k]))+'\n')

with open('out-PALM-AT-eps-fixed-v2-PSNR.txt', 'w') as f:
    f.write('sig max mean min \n')
    for k in range(len(tab_std)):
        f.write(str(tab_std[k])+' '+str(np.max(out_PALM_AT_eps_fixed_PSNR[:,k]))+' '+str(np.mean(out_PALM_AT_eps_fixed_PSNR[:,k]))+' '+str(np.min(out_PALM_AT_eps_fixed_PSNR[:,k]))+'\n')

with open('out-PALM-l1-v2-PSNR.txt', 'w') as f:
    f.write('sig max mean min \n')
    for k in range(len(tab_std)):
        f.write(str(tab_std[k])+' '+str(np.max(out_PALM_l1_PSNR[:,k]))+' '+str(np.mean(out_PALM_l1_PSNR[:,k]))+' '+str(np.min(out_PALM_l1_PSNR[:,k]))+'\n')



# labels = ['0.02', '0.05','0.1','0.2']

# plt.figure(figsize=(20,10))
# plt.plot(out_TROF.mean(axis=0),label='TROF')
# plt.plot(out_PALM_l1.mean(axis=0),label='PALM l1')
# plt.plot(out_PALM_AT_eps_descent.mean(axis=0),label='PALM AT eps descent')
# plt.plot(out_PALM_AT_eps_fixed.mean(axis=0),label='PALM AT eps fixed')
# plt.plot(out_Hohm.mean(axis=0),label='Hohm')
# plt.xticks([0, 1, 2,3],labels)
# plt.legend()
# plt.savefig('boxplot-blur-Jaccard.png')
# plt.show()
# plt.figure(figsize=(20,10))
# plt.plot(out_TROF_PSNR.mean(axis=0),label='TROF')
# plt.plot(out_PALM_l1_PSNR.mean(axis=0),label='PALM l1')
# plt.plot(out_PALM_AT_eps_descent_PSNR.mean(axis=0),label='PALM AT eps descent')
# plt.plot(out_PALM_AT_eps_fixed_PSNR.mean(axis=0),label='PALM AT eps fixed')
# plt.plot(out_Hohm_PSNR.mean(axis=0),label='Hohm')
# plt.xticks([0, 1, 2,3],labels)
# plt.legend()
# plt.savefig('boxplot-blur-PSNR.png')
# plt.show()
