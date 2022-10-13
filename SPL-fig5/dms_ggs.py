import sys
sys.path.insert(0, 'lib/')

from dms import *
from tools_dms import *
from tools_trof import *
from PIL import Image
import scipy as scp
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy.io

# namefile = 'dots_52_v3_noise_0.05_blur_1_2_1'
# namefile = 'dots_52_v3_noise_0.02_blur_3_2_1'
# namefile = 'dots_52_v3_noise_0.08_blur_3_2_1'
# namefile = 'dots_52_v3_noise_0.08_blur_11_1_1'

def dms(namefile,method,normtype,mit=300,eps=2.):
    degraded_data=  scipy.io.loadmat('../degraded_images/'+namefile+'.mat')
    f = degraded_data['f']
    fNoisy = degraded_data['fNoisy']
    e_exacte = degraded_data['e_exacte']
    A = degraded_data['A_python']
    eps_min=0
    if method =='PALM-eps-descent':
        eps_min = 0.02
        # mit = int(mit/(eps/eps_min))
    a2,b2,c2,im_rec_palm_l1,cont_rec_palm_l1=golden_section_map(lmin=-6,lmax=0,bmin=-1,bmax=2,
                                                                noised_im1=fNoisy,im1=f,
                                                                contours_im1=e_exacte,scale_type='10',
                                                                stop_crit=1e-4,grid_size=5,max_round=5,
                                                                objective='Jaccard',method=method,norm_type=normtype,
                                                                maxiter=mit,eps=eps,time_limit=360000,blur_type='Gaussian',
                                                                A=A,eps_AT_min=eps_min)
    print(np.max(cont_rec_palm_l1))

    scipy.io.savemat('../results/'+method+'_'+normtype+'_'+str(mit)+'_'+namefile+'.mat', dict(u_rec=im_rec_palm_l1,e_rec=cont_rec_palm_l1))


i=0
def run_series_dms(nameimage,noise_std,blur_size=1,blur_std=1,normtype='l1',numreal=1,method='SL-PAM',mit=300,eps=2.):
    for i in range(numreal):
        dms(nameimage+'_noise_'+str(noise_std)+'_blur_'+str(blur_size)+'_'+str(blur_std)+'_'+str(i+1),method=method,normtype=normtype,mit=mit,eps=eps)

mit=300


#### 0.1#####
run_series_dms('dots_52_v3',0.1,1,1,'l1',10,'SL-PAM',mit=mit)
run_series_dms('dots_52_v3',0.1,1,1,'l1',10,'PALM',mit=mit)
run_series_dms('dots_52_v3',0.1,1,1,'AT-fourier',10,'PALM-eps-descent',mit=mit,eps=2)
run_series_dms('dots_52_v3',0.1,1,1,'AT-fourier',10,'PALM',mit=mit,eps=0.02)


#### 0.02#####
run_series_dms('dots_52_v3',0.02,1,1,'l1',10,'SL-PAM',mit=mit)
run_series_dms('dots_52_v3',0.02,1,1,'l1',10,'PALM',mit=mit)
run_series_dms('dots_52_v3',0.02,1,1,'AT-fourier',10,'PALM-eps-descent',mit=mit,eps=2)
run_series_dms('dots_52_v3',0.02,1,1,'AT-fourier',10,'PALM',mit=mit,eps=0.02)

#### 0.08#####
run_series_dms('dots_52_v3',0.08,1,1,'l1',10,'SL-PAM',mit=mit)
run_series_dms('dots_52_v3',0.08,1,1,'l1',10,'PALM',mit=mit)
run_series_dms('dots_52_v3',0.08,1,1,'AT-fourier',10,'PALM-eps-descent',mit=mit,eps=2)
run_series_dms('dots_52_v3',0.08,1,1,'AT-fourier',10,'PALM',mit=mit,eps=0.02)

#### 0.2#####
run_series_dms('dots_52_v3',0.2,1,1,'l1',10,'SL-PAM',mit=mit)
run_series_dms('dots_52_v3',0.2,1,1,'l1',10,'PALM',mit=mit)
run_series_dms('dots_52_v3',0.2,1,1,'AT-fourier',10,'PALM-eps-descent',mit=mit,eps=2)
run_series_dms('dots_52_v3',0.2,1,1,'AT-fourier',10,'PALM',mit=mit,eps=0.02)

