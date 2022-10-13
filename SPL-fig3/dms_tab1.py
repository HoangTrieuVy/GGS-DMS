import sys
sys.path.insert(0, '../python_dms/lib/')

from dms import *
from tools_dms import *
from tools_trof import *
from PIL import Image
import scipy as scp
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy.io


namefile = 'dots_52_v3_noise_0.05_blur_1_1_3'

degraded_data=  scipy.io.loadmat('../degraded_images/'+namefile+'.mat')
f = degraded_data['f']
fNoisy = degraded_data['fNoisy']
e_exacte = degraded_data['e_exacte']
A = degraded_data['A_python']
eps_min=0
eps_min = 0.02

print(f)
print(fNoisy)

method='PALM'
normtype='AT-fourier'
eps=0.2
# mit=300
# a2,b2,c2,im_rec,cont_rec=golden_section_map(lmin=-6,lmax=0,bmin=-1,bmax=2,
#                                                             noised_im1=fNoisy,im1=f,
#                                                             contours_im1=e_exacte,scale_type='10',
#                                                             stop_crit=1e-4,grid_size=5,max_round=5,
#                                                             objective='Jaccard',method=method,norm_type=normtype,
#                                                             maxiter=mit,eps=eps,time_limit=360000,blur_type='Gaussian',
#                                                             A=A,eps_AT_min=eps_min)
# print('Noisy PSNR:',PSNR(fNoisy,f))
# print('DMS PSNR: ', PSNR(im_rec,f))
# print('Jaccard:',jaccard(cont_rec,e_exacte))

# f=plt.figure(figsize=(8,8))
# plt.axis('off')
# plt.imshow(im_rec,'gray')
# draw_contour(cont_rec,'',fig=f)
# plt.savefig('../SPL-fig3/'+namefile+'_Jaccard_ggs.png', bbox_inches='tight', pad_inches=0)


# draw_dots_multiresolution(b2,a2,beta_axis=np.linspace(-1,2,5),lambda_axis=np.linspace(-6,0,5),name='PSNR')
# plt.savefig('fig1.png')

print('Noisy PSNR:',PSNR(fNoisy,f))
a2,b2,c2,im_rec,cont_rec=golden_section_map(lmin=-6,lmax=0,bmin=-1,bmax=2,
                                                            noised_im1=fNoisy,im1=f,
                                                            contours_im1=e_exacte,scale_type='10',
                                                            stop_crit=1e-4,grid_size=5,max_round=5,
                                                            objective='PSNR',method=method,norm_type=normtype,
                                                            maxiter=3000,eps=eps,time_limit=360000,blur_type='Gaussian',
                                                            A=A,eps_AT_min=eps_min)
print('DMS PSNR: ', PSNR(im_rec,f))
print('Jaccard:',jaccard(cont_rec,e_exacte))

f=plt.figure(figsize=(8,8))
plt.axis('off')
plt.imshow(im_rec,'gray')
draw_contour(cont_rec,'',fig=f)
# plt.savefig('../SPL-fig3/'+namefile+'_PSNR_ggs.png', bbox_inches='tight', pad_inches=0)
draw_dots_multiresolution(b2,a2,beta_axis=np.linspace(-1,2,5),lambda_axis=np.linspace(-6,0,5),name='PSNR')
# plt.savefig('fig2.png')
