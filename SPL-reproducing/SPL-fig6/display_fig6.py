import sys
sys.path.insert(0, '../../python_dms/lib/')

# from dms import *
from tools_dms import *
from tools_trof import *
from PIL import Image
import scipy as scp
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy.io

namefile = ['35010_noise_0.05_blur_1_1_1','24004_noise_0.05_blur_1_1_1','35008_noise_0.05_blur_1_1_1',
'118035_noise_0.05_blur_1_1_1','65019_noise_0.05_blur_1_1_1','35010_noise_0.05_blur_1_1_1','124084_noise_0.05_blur_1_1_1',
'12003_noise_0.05_blur_1_1_1']
name=['35010','24004','35008','118035','65019','35010','124084','12003']

for name,namefile in zip(name,namefile):

	degraded_data=  scipy.io.loadmat('../degraded_images/'+namefile+'.mat')
	f = degraded_data['f']
	fNoisy = degraded_data['fNoisy']
	e_exacte = degraded_data['e_exacte']
	A = degraded_data['A_python']

	ac=plt.figure(figsize=(8,8))
	plt.imshow(fNoisy,'gray',vmin=np.min(fNoisy),vmax=np.max(fNoisy))
	draw_contour(e_exacte,'',fig=ac)
	plt.axis('off')
	plt.savefig('noised_'+name+'.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)

	i=0
	im_rec_Hohm= scipy.io.loadmat('../results/Hohm_'+name+'_'+'noise'+'_'+str(0.05)+'_'+'blur'+'_'+str(1)+'_'+str(1)+'_'+str(i+1)+'.mat')['u_rec']
	cont_rect_Hohm= scipy.io.loadmat('../results/Hohm_'+name+'_'+'noise'+'_'+str(0.05)+'_'+'blur'+'_'+str(1)+'_'+str(1)+'_'+str(i+1)+'.mat')['e_rec']

	ac=plt.figure(figsize=(8,8))
	plt.imshow(im_rec_Hohm,'gray',vmin=np.min(fNoisy),vmax=np.max(fNoisy))
	draw_contour(cont_rect_Hohm,'',fig=ac)
	plt.axis('off')
	plt.savefig('Hohm_'+name+'.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)
	print('PNSR Hohm',PSNR(im_rec_Hohm,f))
	print('Jaccard Hohm',jaccard(cont_rect_Hohm,e_exacte))


	im_rec_AT_descent= scipy.io.loadmat('../results/PALM-eps-descent_AT-fourier_300_'+name+'_'+'noise'+'_'+str(0.05)+'_'+'blur'+'_'+str(1)+'_'+str(1)+'_'+str(i+1)+'.mat')['u_rec']
	cont_rect_AT_descent= scipy.io.loadmat('../results/PALM-eps-descent_AT-fourier_300_'+name+'_'+'noise'+'_'+str(0.05)+'_'+'blur'+'_'+str(1)+'_'+str(1)+'_'+str(i+1)+'.mat')['e_rec']
	ac=plt.figure(figsize=(8,8))
	plt.imshow(im_rec_AT_descent,'gray',vmin=np.min(fNoisy),vmax=np.max(fNoisy))
	draw_contour(cont_rect_AT_descent,'',fig=ac)
	plt.axis('off')
	plt.savefig('PALM_AT_descent_'+name+'.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)

	print('PSNR PALM AT descent', PSNR(im_rec_AT_descent,f))
	print('Jaccard PALM AT descent',jaccard(cont_rect_AT_descent,e_exacte))



	im_rec_l1= scipy.io.loadmat('../results/PALM_l1_300_'+name+'_'+'noise'+'_'+str(0.05)+'_'+'blur'+'_'+str(1)+'_'+str(1)+'_'+str(i+1)+'.mat')['u_rec']
	cont_rect_l1= scipy.io.loadmat('../results/PALM_l1_300_'+name+'_'+'noise'+'_'+str(0.05)+'_'+'blur'+'_'+str(1)+'_'+str(1)+'_'+str(i+1)+'.mat')['e_rec']

	ac=plt.figure(figsize=(8,8))
	plt.imshow(im_rec_l1,'gray',vmin=np.min(fNoisy),vmax=np.max(fNoisy))
	plt.axis('off')
	draw_contour(cont_rect_l1,'',fig=ac)
	plt.savefig('PALM-l1_'+name+'.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)

	print('PSNR PALM l1',PSNR(im_rec_l1,f))
	print('Jaccard PALM l1', jaccard(cont_rect_l1,e_exacte))


	im_rec_TROF= scipy.io.loadmat('../results/TROF_300_'+name+'_'+'noise'+'_'+str(0.05)+'_'+'blur'+'_'+str(1)+'_'+str(1)+'_'+str(i+1)+'.mat')['u_rec']
	cont_rect_TROF= scipy.io.loadmat('../results/TROF_300_'+name+'_'+'noise'+'_'+str(0.05)+'_'+'blur'+'_'+str(1)+'_'+str(1)+'_'+str(i+1)+'.mat')['e_rec']
	ac=plt.figure(figsize=(8,8))
	plt.imshow(im_rec_TROF,'gray',vmin=np.min(fNoisy),vmax=np.max(fNoisy))
	draw_contour(cont_rect_TROF,'',fig=ac)
	plt.axis('off')
	plt.savefig('TROF_'+name+'.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)

	print('PSNR TROF', PSNR(im_rec_TROF,f))
	print('Jaccard TROF',jaccard(cont_rect_TROF,e_exacte))