import sys
sys.path.insert(0, '../../python_dms/lib/')
from tools_dms import *
from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.io


namefile = 'dots_52_v3_noise_0.05_blur_1_1_3'
degraded_data=  scipy.io.loadmat('../../degraded_images/'+namefile+'.mat')
f = degraded_data['f']
fNoisy = degraded_data['fNoisy']
e_exacte = degraded_data['e_exacte']
A = degraded_data['A_python']
eps_min=0.02
niter=300
name= 'dots_52_v3'
mit =300
i=0


############# PALM AT eps descent ######################
# start =time.time()
a2,b2,c2,im_rec,cont_rec=golden_section_map(lmin=-6,lmax=0,bmin=-1,bmax=2,
                                            noised_im1=fNoisy,im1=f,
                                            contours_im1=e_exacte,scale_type='10',
                                            stop_crit=1e-4,grid_size=5,max_round=5,
                                            objective='Jaccard',method='PALM-eps-descent',norm_type='AT',
                                            maxiter=mit,eps=2.,
                                            A=A,eps_AT_min=eps_min)

# print('PALM AT descent CT ',time.time()-start)
ac=plt.figure(figsize=(8,8))
plt.axis('off')
plt.imshow(im_rec,'gray')
draw_contour(cont_rec,'',fig=ac)

# plt.savefig('PALM-AT-descent.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)


# print('PSNR PALM AT descent',PSNR(im_rec,f))
# print('Jaccard PALM At descent', jaccard(cont_rec,e_exacte))


# ########### Mumford Shah 2D Hohm et al.################
# im_rec_Hohm= scipy.io.loadmat('../../results/Hohm_'+name+'_'+'noise'+'_'+str(0.05)+'_'+'blur'+'_'+str(1)+'_'+str(1)+'_'+str(i+1)+'.mat')['u_rec']
# cont_rect_Hohm= scipy.io.loadmat('../../results/Hohm_'+name+'_'+'noise'+'_'+str(0.05)+'_'+'blur'+'_'+str(1)+'_'+str(1)+'_'+str(i+1)+'.mat')['e_rec']

# ac=plt.figure(figsize=(8,8))
# plt.imshow(im_rec_Hohm,'gray')
# draw_contour(cont_rect_Hohm,'',fig=ac)
# plt.axis('off')
# plt.savefig('Hohm.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)
# print('PNSR Hohm',PSNR(im_rec_Hohm,f))
# print('Jaccard Hohm',jaccard(cont_rect_Hohm,e_exacte))



# ##################### TV 2D - T-ROF########################
# start =time.time()
# a0,b0,im_rec_TROF,cont_rect_TROF = grid_search_TROF(fNoisy,f,K_min=2,K_max=6,chi_min=-2,chi_max=2,maxiter=mit,contours_im1=e_exacte)
# ac=plt.figure(figsize=(8,8))
# plt.imshow(im_rec_TROF,'gray')
# draw_contour(cont_rect_TROF,'',fig=ac)
# plt.axis('off')
# # plt.savefig('TROF.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)
# print('TROF CT ',time.time()-start)
# print('PSNR TROF', PSNR(im_rec_TROF,f))
# print('Jaccard TROF',jaccard(cont_rect_TROF,e_exacte))



# ################### PALM-l1############################
# start =time.time()
# a2,b2,c2,im_rec,cont_rec=golden_section_map(lmin=-6,lmax=0,bmin=-1,bmax=2,
#                                                                noised_im1=fNoisy,im1=f,
#                                                                contours_im1=e_exacte,scale_type='10',
#                                                                stop_crit=1e-4,grid_size=5,max_round=5,
#                                                                objective='Jaccard',method='PALM',norm_type='l1',
#                                                                maxiter=mit,eps=0.2,blur_type='Gaussian',
#                                                                A=A,eps_AT_min=eps_min)

# print('PALM l1 CT ',time.time()-start)
# ac=plt.figure(figsize=(8,8))
# plt.imshow(im_rec,'gray')
# plt.axis('off')
# draw_contour(cont_rec,'',fig=ac)
# # plt.savefig('PALM-l1.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)

# print('PSNR PALM l1',PSNR(im_rec,f))
# print('Jaccard PALM l1', jaccard(cont_rec,e_exacte))



# ######################## PALM -AT -epsilon fixed = 0.2######################
# start =time.time()
# a2,b2,c2,im_rec,cont_rec=golden_section_map(lmin=-6,lmax=0,bmin=-1,bmax=2,
#                                                                noised_im1=fNoisy,im1=f,
#                                                                contours_im1=e_exacte,scale_type='10',
#                                                                stop_crit=1e-4,grid_size=5,max_round=5,
#                                                                objective='Jaccard',method='PALM',norm_type='AT-fourier',
#                                                                maxiter=mit,eps=0.2,blur_type='Gaussian',
#                                                                A=A,eps_AT_min=eps_min)

# print('PALM AT fixed CT  ',time.time()-start)
# ac=plt.figure(figsize=(8,8))
# plt.axis('off')
# plt.imshow(im_rec,'gray')
# draw_contour(cont_rec,'',fig=ac)
# # plt.savefig('PALM-AT-fixed.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)



# print('PSNR PALM AT fixed',PSNR(im_rec,f))
# print('Jaccard PALM AT fixed', jaccard(cont_rec,e_exacte))


# ################# SL PAM l1##############################
# start = time.time()
# a2,b2,c2,im_rec,cont_rec=golden_section_map(lmin=-6,lmax=0,bmin=-1,bmax=2,
#                                                                 noised_im1=fNoisy,im1=f,
#                                                                 contours_im1=e_exacte,scale_type='10',
#                                                                 stop_crit=1e-4,grid_size=5,max_round=5,
#                                                                 objective='Jaccard',method='SL-PAM',norm_type='l1',
#                                                                 maxiter=50,eps=2,blur_type='Gaussian',
#                                                                 A=A,eps_AT_min=eps_min)
# print('SL PAM l1 CT :', time.time()-start)
# ac=plt.figure(figsize=(8,8))
# plt.axis('off')
# plt.imshow(im_rec,'gray')
# draw_contour(cont_rec,'',fig=ac)
# # plt.savefig('SLPAM-l1.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)
# # np.save('im_rec_SLPAM_l1.npy',im_rec)
# # np.save('cont_rec_SLPAM_l1.npy',cont_rec)
# # np.savetxt('jaccard_PSNR_CT_SLPAM_l1.out',np.array([jaccard(cont_rec,e_exacte),PSNR(im_rec,f),time.time()-start]))
# print('PSNR SLPAM l1',PSNR(im_rec,f))
# print('Jaccard SLPAM l1',jaccard(cont_rec,e_exacte))


# ##################### SLPAM - AT epsilon fixed=0.2 #####################
# start =time.time()
# a2,b2,c2,im_rec,cont_rec=golden_section_map(lmin=-6,lmax=0,bmin=-1,bmax=2,
#                                                                 noised_im1=fNoisy,im1=f,
#                                                                 contours_im1=e_exacte,scale_type='10',
#                                                                 stop_crit=1e-4,grid_size=5,max_round=5,
#                                                                 objective='Jaccard',method='SL-PAM',norm_type='AT',
#                                                                 maxiter=50,eps=0.2,blur_type='Gaussian',
#                                                                 A=A,eps_AT_min=eps_min)

# print('SL PAM AT fixed CT ',time.time()-start)
# ac=plt.figure(figsize=(8,8))
# plt.axis('off')
# plt.imshow(im_rec,'gray')
# draw_contour(cont_rec,'',fig=ac)
# # plt.savefig('SLPAM-fixed.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)

# # np.save('im_rec_SLPAM_AT_fixed.npy',im_rec)
# # np.save('cont_rec_SLPAM_AT_fixed.npy',cont_rec)
# # np.savetxt('jaccard_PSNR_CT_SLPAM_AT_fixed.out',np.array([jaccard(cont_rec,e_exacte),PSNR(im_rec,f),time.time()-start]))
# print('PSNR SLPAM AT fixed',PSNR(im_rec,f))
# print('Jaccard SLPAM AT fixed', jaccard(cont_rec,e_exacte))


# ########################### SLPAM AT epsilon descent ################################
# start = time.time()
# a2,b2,c2,im_rec,cont_rec=golden_section_map(lmin=-6,lmax=0,bmin=-1,bmax=2,
#                                             noised_im1=fNoisy,im1=f,
#                                             contours_im1=e_exacte,scale_type='10',
#                                             stop_crit=1e-4,grid_size=5,max_round=5,
#                                             objective='Jaccard',method='SLPAM-eps-descent',norm_type='AT',
#                                             maxiter=300,eps=2.,
#                                             A=A,eps_AT_min=eps_min)
# print('SL PAM AT descent CT :', time.time()-start)
# ac=plt.figure(figsize=(8,8))
# plt.axis('off')
# plt.imshow(im_rec,'gray')
# draw_contour(cont_rec,'',fig=ac)
# # plt.savefig('SLPAM-descent.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)

# plt.show()
# print('PSNR SLPAM AT descent',PSNR(im_rec,f))
# print('Jaccard SLPAM AT descent', jaccard(cont_rec,e_exacte))


# np.save('im_rec_SLPAM_AT_descent.npy',im_rec)
# np.save('cont_rec_SLPAM_AT_descent.npy',cont_rec)
# np.savetxt('jaccard_PSNR_CT_SLPAM_AT_descent.out',np.array([jaccard(cont_rec,e_exacte),PSNR(im_rec,f),time.time()-start]))
