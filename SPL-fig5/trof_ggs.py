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



def trof_ggs(namefile,mit=300):
    degraded_data=  scipy.io.loadmat('../degraded_images/'+namefile+'.mat')
    f = degraded_data['f']
    fNoisy = degraded_data['fNoisy']
    e_exacte = degraded_data['e_exacte']

    a0,b0,im_rec_TROF,cont_rect_TROF = grid_search_TROF(fNoisy,f,K_min=2,K_max=6,chi_min=-2,chi_max=2,maxiter=mit,contours_im1=e_exacte)


    scipy.io.savemat('../results/'+'TROF'+'_'+str(mit)+'_'+namefile+'.mat', dict(u_rec=im_rec_TROF,e_rec=cont_rect_TROF))



trof_ggs('dots_52_v3_noise_0.08_blur_1_1_1')
trof_ggs('dots_52_v3_noise_0.08_blur_1_1_2')
trof_ggs('dots_52_v3_noise_0.08_blur_1_1_3')
trof_ggs('dots_52_v3_noise_0.08_blur_1_1_4')
trof_ggs('dots_52_v3_noise_0.08_blur_1_1_5')
trof_ggs('dots_52_v3_noise_0.08_blur_1_1_6')
trof_ggs('dots_52_v3_noise_0.08_blur_1_1_7')
trof_ggs('dots_52_v3_noise_0.08_blur_1_1_8')
trof_ggs('dots_52_v3_noise_0.08_blur_1_1_9')
trof_ggs('dots_52_v3_noise_0.08_blur_1_1_10')


trof_ggs('dots_52_v3_noise_0.02_blur_1_1_1')
trof_ggs('dots_52_v3_noise_0.02_blur_1_1_2')
trof_ggs('dots_52_v3_noise_0.02_blur_1_1_3')
trof_ggs('dots_52_v3_noise_0.02_blur_1_1_4')
trof_ggs('dots_52_v3_noise_0.02_blur_1_1_5')
trof_ggs('dots_52_v3_noise_0.02_blur_1_1_6')
trof_ggs('dots_52_v3_noise_0.02_blur_1_1_7')
trof_ggs('dots_52_v3_noise_0.02_blur_1_1_8')
trof_ggs('dots_52_v3_noise_0.02_blur_1_1_9')
trof_ggs('dots_52_v3_noise_0.02_blur_1_1_10')

trof_ggs('dots_52_v3_noise_0.1_blur_1_1_1')
trof_ggs('dots_52_v3_noise_0.1_blur_1_1_2')
trof_ggs('dots_52_v3_noise_0.1_blur_1_1_3')
trof_ggs('dots_52_v3_noise_0.1_blur_1_1_4')
trof_ggs('dots_52_v3_noise_0.1_blur_1_1_5')
trof_ggs('dots_52_v3_noise_0.1_blur_1_1_6')
trof_ggs('dots_52_v3_noise_0.1_blur_1_1_7')
trof_ggs('dots_52_v3_noise_0.1_blur_1_1_8')
trof_ggs('dots_52_v3_noise_0.1_blur_1_1_9')
trof_ggs('dots_52_v3_noise_0.1_blur_1_1_10')


trof_ggs('dots_52_v3_noise_0.2_blur_1_1_1')
trof_ggs('dots_52_v3_noise_0.2_blur_1_1_2')
trof_ggs('dots_52_v3_noise_0.2_blur_1_1_3')
trof_ggs('dots_52_v3_noise_0.2_blur_1_1_4')
trof_ggs('dots_52_v3_noise_0.2_blur_1_1_5')
trof_ggs('dots_52_v3_noise_0.2_blur_1_1_6')
trof_ggs('dots_52_v3_noise_0.2_blur_1_1_7')
trof_ggs('dots_52_v3_noise_0.2_blur_1_1_8')
trof_ggs('dots_52_v3_noise_0.2_blur_1_1_9')
trof_ggs('dots_52_v3_noise_0.2_blur_1_1_10')
