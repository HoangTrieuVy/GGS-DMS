import sys
sys.path.insert(0, 'lib/')
sys.path.insert(0, 'comparison_lib/')

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
# namefile = 'dots_52_v3_noise_0.08_blur_11_1_1'

def trof_ggs(namefile,mit=300):
    degraded_data=  scipy.io.loadmat('../degraded_images/'+namefile+'.mat')
    f = degraded_data['f']
    fNoisy = degraded_data['fNoisy']
    e_exacte = degraded_data['e_exacte']

    a0,b0,im_rec_TROF,cont_rect_TROF = grid_search_TROF(fNoisy,f,K_min=2,K_max=6,chi_min=-2,chi_max=2,maxiter=mit,contours_im1=e_exacte)


    scipy.io.savemat('../results/'+'TROF'+'_'+str(mit)+'_'+namefile+'.mat', dict(u_rec=im_rec_TROF,e_rec=cont_rect_TROF))

namefile = '124084_noise_0.05_blur_1_1_1'
trof_ggs(namefile)


namefile = '35010_noise_0.05_blur_1_1_1'
trof_ggs(namefile)


namefile = '65019_noise_0.05_blur_1_1_1'
trof_ggs(namefile)

namefile = '118035_noise_0.05_blur_1_1_1'
trof_ggs(namefile)


namefile = '35008_noise_0.05_blur_1_1_1'
trof_ggs(namefile)


namefile = '24004_noise_0.05_blur_1_1_1'
trof_ggs(namefile)

namefile = '12003_noise_0.05_blur_1_1_1'
trof_ggs(namefile)

