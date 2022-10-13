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
    elif method = 'PALM':
        mit = 1500
    a2,b2,c2,im_rec_palm_l1,cont_rec_palm_l1=golden_section_map(lmin=-6,lmax=0,bmin=-1,bmax=2,
                                                                noised_im1=fNoisy,im1=f,
                                                                contours_im1=e_exacte,scale_type='10',
                                                                stop_crit=1e-4,grid_size=5,max_round=5,
                                                                objective='Jaccard',method=method,norm_type=normtype,
                                                                maxiter=mit,eps=eps,time_limit=360000,blur_type='Gaussian',
                                                                A=A,eps_AT_min=eps_min)

    scipy.io.savemat('../results/'+method+'_'+normtype+'_'+str(mit)+'_'+namefile+'.mat', dict(u_rec=im_rec_palm_l1,e_rec=cont_rec_palm_l1))

#### 0.05#####

namefile = '12003_noise_0.05_blur_1_1_1'
dms(namefile,method='PALM',normtype='l1')#CT 1067
dms(namefile,method='PALM-eps-descent',normtype='AT-fourier',eps=2.) #CT 61770

# namefile = '124084_noise_0.05_blur_1_1_1'
# dms(namefile,method='PALM',normtype='l1')#CT 1112
# dms(namefile,method='PALM-eps-descent',normtype='AT-fourier',eps=2.)#CT 57555

# namefile = '35010_noise_0.05_blur_1_1_1'
# dms(namefile,method='PALM',normtype='l1')#CT 988
# dms(namefile,method='PALM-eps-descent',normtype='AT-fourier',eps=2.)#CT 49888

# namefile = '65019_noise_0.05_blur_1_1_1'
# dms(namefile,method='PALM',normtype='l1')#CT 1222
# dms(namefile,method='PALM-eps-descent',normtype='AT-fourier',eps=2.)#CT 60059

# namefile = '118035_noise_0.05_blur_1_1_1'
# dms(namefile,method='PALM',normtype='l1')#CT 1221
# dms(namefile,method='PALM-eps-descent',normtype='AT-fourier',eps=2.)#CT 59009

# namefile = '35008_noise_0.05_blur_1_1_1'
# dms(namefile,method='PALM',normtype='l1')#CT 1002
# dms(namefile,method='PALM-eps-descent',normtype='AT-fourier',eps=2.)#CT 52992

# namefile = '24004_noise_0.05_blur_1_1_1'
# dms(namefile,method='PALM',normtype='l1')#CT 1099
# dms(namefile,method='PALM-eps-descent',normtype='AT-fourier',eps=2.)#CT 51002