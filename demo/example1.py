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


namefile = '12003_noise_0.05_blur_1_1_1'

degraded_data=  scipy.io.loadmat('../degraded_images/'+namefile+'.mat')
f = degraded_data['f']
fNoisy = degraded_data['fNoisy']
e_exacte = degraded_data['e_exacte']
A = degraded_data['A_python']
eps_min=0
eps_min = 0.02


method='PALM'
normtype='l1'
eps=0.2
mit=300
# a2,b2,c2,im_rec,cont_rec=golden_section_map(lmin=-6,lmax=0,bmin=-1,bmax=2,
#                                                             noised_im1=fNoisy,im1=f,
#                                                             contours_im1=e_exacte,scale_type='10',
#                                                             stop_crit=1e-4,grid_size=5,max_round=5,
#                                                             objective='Jaccard',method=method,norm_type=normtype,
#                                                             maxiter=mit,eps=eps,time_limit=360000,blur_type='Gaussian',
#                                                             A=A,eps_AT_min=eps_min)

test = DMS(f, '', noise_type='Gaussian',blur_type='Gaussian',
                               beta=2, lamb=0.001, method=method,MaximumIteration=mit,
                               noised_image_input=fNoisy, norm_type=normtype,stop_criterion=1e-4, dkSLPAM=1e-4,
                               optD='OptD',eps=eps,time_limit=360000,A=A)

out = test.process()
print('Noisy PSNR:',PSNR(fNoisy,f))
print('DMS PSNR: ', PSNR(out[1],f))
print('Jaccard:',jaccard(out[0],e_exacte))

fig=plt.figure(figsize=(8,8))
plt.axis('off')
plt.imshow(out[1],'gray')
draw_contour(out[0],'',fig=fig)
plt.savefig(namefile+'_out.png', bbox_inches='tight', pad_inches=0)

fig=plt.figure(figsize=(8,8))
plt.axis('off')
plt.imshow(fNoisy,'gray')
plt.savefig(namefile+'.png', bbox_inches='tight', pad_inches=0)

fig=plt.figure(figsize=(8,8))
plt.axis('off')
plt.imshow(f,'gray')
draw_contour(e_exacte,'',fig=fig)
plt.savefig(namefile+'_exact'+'.png', bbox_inches='tight', pad_inches=0)
