    # -*- coding: utf-8 -*-
import numpy as np
from scipy import fftpack
from scipy import linalg
import time
from scipy.sparse import diags
import scipy as scp
from scipy.sparse import hstack
from scipy.sparse import vstack
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy import signal
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class DMS():
    def __init__(self,save_name,blur_type='none',blur_size=5,blur_std=2,
        noise_peak=30,noise_std=0.1,noise_type='Gaussian',norm_type='l1',edges='similar',
        beta=8,lamb=1e-2,eps=0.2,stop_criterion=1e-4,MaximumIteration=5000,alphabeta=True,
        method='SL-PAM',draw='line',noised_image_input=None,itadd=0,optD='OptD',
        dkSLPAM=1e-4,type_contour='zeros',type_u0='Z',eps_AT_min=0.02,epsilon_log_descent=False,e_init=None,
        gamma_1= 1e-2,gamma_2 = 1e-2,time_limit=3600,A=None):


        self.epsilon_log_descent = epsilon_log_descent
        if(self.epsilon_log_descent is True):
            self.tab_eps = eps*np.exp(-0.5*np.linspace(0,10,MaximumIteration))

        self.e_init = e_init
        self.alphabeta = alphabeta
        self.time_limit= time_limit
        self.optD_type= optD
        self.type_u0=type_u0
        self.type_contour = type_contour
        self.itadd=itadd
        self.dim_e = None
        self.eps = eps
        self.eps_AT_min =eps_AT_min
        self.MaximumIteration = MaximumIteration
        self.stop_criterion = stop_criterion
        self.dk_SLPAM = dkSLPAM
        self.noised_image_input = noised_image_input
        self.draw = draw
        self.method= method
        self.save_name = save_name
        self.beta = beta
        self.lam = lamb
        self.sig = noise_std
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.blur_type = blur_type
        self.blur_size = blur_size
        self.blur_std = blur_std
        self.noise_peak = noise_peak
        self.norm_type = norm_type
        self.edges = edges

        if self.norm_type =='l1l2':
            self.gamma_1 = gamma_1
            self.gamma_2 = gamma_2


        #SL-PAM parameters
        self.error_curve = None
        self.ck_SLPAM = None
        self.dk_SLPAM = dkSLPAM
        self.Jn_SLPAM = None
        self.un_SLPAM = None
        self.en_SLPAM = None
        self.time_SLPAM= None 
        self.A = A
        self.blur_op = None
        self.Aadjoint = None
        self.J_initial = 0
        self.error_image_SLPAM = None
        self.image_reconstructed= None
        self.norms = CreateNorms(self.eps)
        self.prox = ProximityOperators(self.eps)
        self.it_SLPAM =None
        self.gif_SLPAM= []
        self.gif_contour_SLPAM=[]

        #PALM parameter
        self.ck_PALM = None
        self.dk_PALM = None
        self.time_PALM = None
        self.image_reconstructed_PALM = None
        self.norms2 = CreateNorms(self.eps)
        self.prox2 = ProximityOperators(self.eps)
        self.error_curve_PALM = None
        self.Jn_PALM = None
        self.un_PALM = None
        self.en_PALM = None
        self.error_image_PALM = None
        self.it_PALM = None
        self.gif_PALM=[]
        self.gif_contour_PALM= []

        #Image variable
        shape = np.shape(noised_image_input)
        self.size = np.size(shape)
        if self.size == 2:
            self.rows, self.cols = noised_image_input.shape
            self.canal = 1
            if np.max(noised_image_input)>100:
                self.image = (np.copy(noised_image_input))/255.
            else:
                self.image = (np.copy(noised_image_input))
        elif self.size ==3 :
            self.rows, self.cols, self.canal= noised_image_input.shape
            if np.max(noised_image_input)>100:
                self.image = (np.copy(noised_image_input))/255.
            else:
                self.image = (np.copy(noised_image_input))
            self.canal = 3
            self.canal0 = self.image[:,:,0]
            self.canal1 = self.image[:, :, 1]
            self.canal2 = self.image[:, :, 2]
        self.rowscols = self.rows*self.cols
        self.image_degraded = None
        self.stop_descent_contour=0.1*(2*self.rows*self.cols-self.rows-self.cols)


        #Operator
        self.H= self.opt_H()
        self.V= self.opt_V()
        self.optD0= self.optD0_create()
        self.optD1 = self.optD1_create()
        self.mat = self.optD0.dot(self.optD0.T)

        # kernel2 = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        kernel2 = np.array([[1],[-1]])
        kernel1 = np.array([[1,-1]])
        # kernel1 = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        self.h = psf2otf(kernel1,(self.rows,self.cols))
        self.v = psf2otf(kernel2,(self.rows,self.cols))

    def normalization(self,x):
        z = (x - np.min(x)) / (np.max(x) - np.min(x))
        return z
    def Addblur(self):
        if self.blur_type == 'Gaussian':
            # fs = (self.blur_size - 1) / 2.
            # x = np.arange(-fs, fs+1)
            # y = np.arange(-fs, fs+1)
            # X, Y = np.meshgrid(x, y)
            # bump = np.exp(-1 / (2 * self.blur_std ** 2) * (X ** 2 + Y ** 2))
            # kernel = bump / np.sum(bump)

            kernel = matlab_style_gauss2D((self.blur_size,self.blur_size),self.blur_std)

            if self.canal==3:
                self.A = psf2otf(kernel, (self.rows, self.cols)).astype('complex128')
                self.A = np.repeat(self.A[:,:,np.newaxis],self.canal,axis=2)
            elif self.canal ==1:
                self.A = psf2otf(kernel, (self.rows, self.cols)).astype('complex128')
            img_ft = fftpack.fft2(self.image, axes=(0, 1))
            img_ft2 = self.A*img_ft
            self.image_degraded = np.real(fftpack.ifft2(img_ft2, axes=(0, 1)))
            self.image_degraded = self.image_degraded

        elif self.blur_type == 'Uniform':
            kernel = np.ones((self.blur_size, self.blur_size))
            kernel = kernel / np.sum(kernel)
            self.A = psf2otf(kernel, (self.rows, self.cols)).astype('complex128')
            if self.canal == 3:
                self.A = np.repeat(self.A[:, :, np.newaxis], self.canal, axis=2)
            img_ft = fftpack.fft2(self.image, axes=(0, 1))
            img_ft2 = self.A * img_ft
            self.image_degraded = np.real(fftpack.ifft2(img_ft2, axes=(0, 1)))
            self.image_degraded =self.image_degraded
        elif self.blur_type=='none':
            self.image_degraded = np.copy(self.image)

    def Aadjoint_process(self, im):
        img_ft = fftpack.fft2(im, axes=(0, 1))
        img_ft2 = np.conj(self.A) * img_ft
        self.Aadjoint = np.real(fftpack.ifft2(img_ft2, axes=(0, 1)))
        self.Aadjoint = self.Aadjoint
        return self.Aadjoint

    def AddNoise(self):
        if self.noise_type == 'Gaussian':
            self.image_degraded = self.image_degraded + (self.noise_std)*np.random.normal(0,1,np.shape(self.image))
            self.image_degraded = self.image_degraded
        elif self.noise_type == 'Poisson':
            poissonNoise = np.random.poisson(self.image_degraded*self.noise_peak)
            self.image_degraded = self.image_degraded + poissonNoise
            self.image_degraded = self.image_degraded
        elif self.noise_type =='none':
            self.image_degraded = np.copy(self.image_degraded)
            self.image_degraded = self.image_degraded


    def perimeter_estimation(self,norm_type,method,en_SLPAM,en_PALM):
        if norm_type=='l1':
            if method=='PALM':
                return np.sum(np.abs(en_PALM[:,:,0]/self.rows)+np.abs(en_PALM[:,:,1]/self.cols))
            elif method == 'SL-PAM':
                return np.sum(np.abs(en_SLPAM[:,:,0]/self.rows)+np.abs(en_SLPAM[:,:,1]/self.cols))
        if norm_type=='AT' or norm_type=='AT-fourier':
            if method=='PALM':
                e = en_PALM
            elif method == 'SL-PAM':
                e = en_SLPAM
            optD1 = self.optD1_create()
            e_ravel_0 = e[:,:,0].ravel('C')/self.rows
            e_ravel_1 = e[:,:,1].ravel('C')/self.cols
            e_ravel = np.hstack((e_ravel_0,e_ravel_1))
            optD1e = optD1.dot(e_ravel)
            return self.eps * self.norms.L2(optD1e) ** 2 + (0.25 / self.eps) * self.norms.L2(e_ravel) ** 2

    def degrade_matrix(self, u):
        img_ft2 = self.A * fftpack.fft2(u, axes=(0, 1))
        Au = np.real(fftpack.ifft2(img_ft2, axes=(0, 1)))
        Au = Au
        return Au

    def L_term(self,u,z):
        if self.noise_type== 'Gaussian' or self.noise_type=='none':
            if self.blur_type =='none':
                a= 0.5*self.norms.L2(u-z)**2.
                return  a
            else:
                temp = self.degrade_matrix(u)-z
                return 0.5*(self.norms.L2(temp))**2.

        elif self.noise_type=='Poisson':
            return self.KLD(u,z,sig=self.sig)
    def L_prox(self,x,tau,z):
        if self.noise_type== 'Gaussian' or self.noise_type=='none':
            if self.blur_type =='none':
                return self.prox.L2(x,tau,z)
            else:
                return self.prox.L2_restoration(x,tau,z,self.A.astype('complex128'))
        elif self.noise_type=='Poisson':
            return self.prox.KLD(x,z,sigma=self.sig,tau=tau)

    def S_term(self,u,e):
        if self.canal==3:
            return np.sum((np.repeat((1-e)[:,:,:,np.newaxis],self.canal//self.dim_e[1],axis=3)*self.optD(u))**2)
        elif self.canal==1:
            return np.sum(((1 - e) * self.optD(u)) ** 2)
    def S_du(self,u,e):
        temp=0
        if self.canal == 3:
            temp = np.repeat(((1-e)**2)[:,:,:,np.newaxis],self.canal,axis=3)
        elif self.canal==1:
            temp =(1 - e) ** 2


        temp = temp * self.optD(u)
        return 2*self.Dadjoint(temp)


    def S_de(self,u,e):

        return -2*(1-e)*self.S_D(u)

    def S_D(self,u):
        if self.canal==3:
            temp = np.sum(self.optD(u)**2,3)
            return temp
        elif self.canal==1:
            temp = self.optD(u) **2
            return temp

    # contour length penalization
    def R_term(self,e):

        if self.norm_type== 'l0':
            return self.norms.L0(e)
        elif self.norm_type=='l1':
            return self.norms.L1(e)
        elif self.norm_type=='l1q':
            return self.norms.quadL1(e)
        elif self.norm_type == 'AT' or self.norm_type == 'AT-fourier':
            optD1 = self.optD1_create()
            e_ravel_0 = e[:,:,0].ravel('C')
            e_ravel_1 = e[:,:,1].ravel('C')
            e_ravel = np.hstack((e_ravel_0,e_ravel_1))
            optD1e = optD1.dot(e_ravel)
            return self.norms.AT(optD1x=optD1e,x=e)
        elif self.norm_type == 'l1l2':
            return self.gamma_1*self.norms.L1(e)+self.gamma_2*self.norms.L2D(e)

    def R_prox(self,e,tau):
        if self.norm_type== 'l0':
            return self.prox.L0(e, tau)
        elif self.norm_type=='l1':
            return self.prox.L1(e,tau)
        elif self.norm_type=='l1q':
            return self.prox.quadL1(e,tau)
        elif self.norm_type=='AT':
            

            e_ravel_0 = e[:,:,0].ravel('C')
            e_ravel_1 = e[:,:,1].ravel('C')
            e_ravel = np.hstack((e_ravel_0,e_ravel_1))
            # e_ravel = csr_matrix(e_ravel)
            C1 = np.eye(self.rowscols*2)*(1+(2*self.lam/(4*self.eps*self.lam/tau)))
            C2 = np.ones((self.rowscols*2,self.rowscols*2))*(2*self.lam*self.eps*self.lam/tau)
            mat = csc_matrix(identity(self.rowscols*2)*C1)+csc_matrix(C2*self.optD1.T.dot(self.optD1))
            inv_mat = scp.sparse.linalg.inv(mat)
            
            temp = inv_mat.dot(e_ravel)#.todense()
            # temp = spsolve(mat,).todense()
            # print(temp.shape)
            e[:,:,0] = temp[:self.rowscols].reshape(self.rows,self.cols)
            e[:,:,1] = temp[self.rowscols:].reshape(self.rows,self.cols)

            return e

        elif self.norm_type=='AT-fourier':
            #tau is lambda/dk
            C1 = 1 + (tau/(2*self.eps))
            C2 = 2*tau*self.eps

            F = 1/(C2*np.conj(self.v)*self.v+C1)
            G = F*C2*np.conj(self.v)*self.h
            J = C2*np.conj(self.h)*self.h+C1
            K = C2*np.conj(self.h)*self.v*G
            L = C2*np.conj(self.h)*self.v
            e[:,:,0] = np.real(fftpack.ifft2( (F+G/(J-K)*L)*fftpack.fft2(e[:,:,0],axes=(0,1)) + G/(J-K)*fftpack.fft2(e[:,:,1],axes=(0,1)),axes=(0,1)))
            e[:,:,1] = np.real(fftpack.ifft2( 1/(J-K)*L*fftpack.fft2(e[:,:,0] , axes=(0,1)) + 1/(J-K)*fftpack.fft2(e[:,:,1],axes=(0,1)),axes=(0,1)))

            return e
        elif self.norm_type =='l1l2':
            return self.prox.L1L2(e,self.gamma_1,self.gamma_2,tau)

    def initialisation_u_e_SLPAM(self,type_contour='zeros'):

        if self.edges == 'similar':
            self.dim_e = [2, 1]
        elif self.edges == 'distinct':
            self.dim_e = [2, np.shape(self.image)[2]]

        if(self.canal==3):
            if type_contour == 'zeros':
                self.en_SLPAM = np.zeros((self.rows, self.cols, self.dim_e[0]))
            elif type_contour == 'ones':
                self.en_SLPAM = np.ones((self.rows,self.cols,self.dim_e[0]))-1e-3
            elif type_contour == 'random01':
                self.en_SLPAM = np.random.uniform(0,1,(self.rows,self.cols,self.dim_e[0]))
        elif(self.canal==1):
            if self.blur_type == 'none':
                if self.type_u0 == 'Z':
                    self.un_SLPAM = np.copy(self.image_degraded)
                elif self.type_u0 == 'random_01':
                    self.un_SLPAM = np.random.uniform(0,1,(self.rows,self.cols))
                elif self.type_u0 == 'denoised_simple':
                    kernel = np.ones((self.blur_size, self.blur_size))
                    kernel = kernel / np.sum(kernel)
                    kernel= psf2otf(kernel, (self.rows, self.cols)).astype('complex128')
                    img_ft = fftpack.fft2(np.copy(self.image_degraded), axes=(0, 1))
                    img_ft2 = kernel* img_ft
                    self.un_SLPAM = np.real(fftpack.ifft2(img_ft2, axes=(0, 1)))
                    self.un_SLPAM = self.un_SLPAM
            elif self.blur_type == 'Gaussian':
                self.un_SLPAM = self.Aadjoint_process(np.copy(self.image_degraded))
            elif self.blur_type=='Uniform':
                self.un_SLPAM = self.Aadjoint_process(np.copy(self.image_degraded))

            if type_contour == 'zeros':
                self.en_SLPAM = np.zeros((self.rows, self.cols, self.dim_e[0]))
            elif type_contour == 'ones':
                self.en_SLPAM = np.ones((self.rows,self.cols,self.dim_e[0]))-1e-3
            elif type_contour == 'random01':
                self.en_SLPAM = np.random.uniform(0,1,(self.rows,self.cols,self.dim_e[0]))
    def initialisation_u_e_PALM(self,type_contour='zeros'):
        if self.edges == 'similar':
            self.dim_e = [2, 1]
        elif self.edges == 'distinct':
            self.dim_e = [2, np.shape(self.image)[2]]

        if (self.canal == 3):
            if type_contour == 'zeros':
                self.en_PALM = np.zeros((self.rows, self.cols, self.dim_e[0]))
            elif type_contour == 'ones':
                self.en_PALM = np.ones((self.rows,self.cols,self.dim_e[0]))
            elif type_contour == 'random01':
                self.en_PALM = np.random.uniform(0,1,(self.rows,self.cols,self.dim_e[0]))

        elif (self.canal == 1):
            if self.blur_type == 'none':
                if self.type_u0 == 'Z':
                    self.un_PALM = self.image_degraded
                elif self.type_u0 == 'random_01':
                    self.un_PALM = np.random.uniform(0,1,(self.rows,self.cols))
                elif self.type_u0 == 'denoised_simple':
                    kernel = np.ones((self.blur_size, self.blur_size))
                    kernel = kernel / np.sum(kernel)
                    kernel= psf2otf(kernel, (self.rows, self.cols)).astype('complex128')
                    img_ft = fftpack.fft2(np.copy(self.image_degraded), axes=(0, 1))
                    img_ft2 = kernel* img_ft
                    self.un_PALM = np.real(fftpack.ifft2(img_ft2, axes=(0, 1)))
                    self.un_PALM = self.un_PALM
            elif self.blur_type == 'Gaussian':
                self.un_PALM = self.Aadjoint_process(np.copy(self.image_degraded))
            elif self.blur_type=='Uniform':
                self.un_PALM = self.Aadjoint_process(np.copy(self.image_degraded))



            if type_contour == 'zeros':
                self.en_PALM = np.zeros((self.rows, self.cols, self.dim_e[0]))
            elif type_contour == 'ones':
                self.en_PALM = np.ones((self.rows,self.cols,self.dim_e[0]))-1e-3
            elif type_contour == 'random01':
                self.en_PALM = np.random.uniform(0,1,(self.rows,self.cols,self.dim_e[0]))
            elif type_contour == 'Z':
                self.en_PALM = self.e_init

    def energy(self,u,e,z):
        self.J = self.L_term(u,z) + self.beta*self.S_term(u,e) + self.lam*self.R_term(e)

        return self.J



    #Difference operator D
    def optD(self, x):

        if self.canal==3:
            y = np.zeros((self.rows, self.cols, 2, self.canal))
            y[:, :, 0, :] = np.concatenate((x[:, 1:, :] - x[:, 0:-1, :], np.zeros((self.rows, 1, self.canal))),axis=1) / 2.
            y[:, :, 1, :] = np.concatenate((x[1:, :, :] - x[0:-1, :, :], np.zeros((1,self.cols, self.canal))),axis=0) / 2.
            return y

        elif self.canal==1:
            y = np.zeros((self.rows, self.cols, 2))
            # # print(temp.shape)
            y[:, :, 0] = np.concatenate((x[:, 1:] - x[:, 0:-1], np.zeros((self.rows, 1))),axis=1) / 2.
            y[:, :, 1] = np.concatenate((x[1:, :] - x[0:-1, :], np.zeros((1, self.cols))),axis=0) / 2.
            # dh = np.array([[-1,1]])/2
            # dv = np.array([[-1],[1]])/2
            # y[:, :, 0] = np.concatenate((signal.convolve2d(x, dh, mode='valid'),np.zeros((self.rows,1))),axis=1)
            # y[:, :, 1] = np.concatenate((signal.convolve2d(x, dv, mode='valid'),np.zeros((1,self.cols))),axis=0)

            return y

    def Dadjoint(self,x):
        if self.canal == 3:
            y1 = np.concatenate((x[:,0,0,:].reshape(self.rows,1,3),x[:,1:-1,0,:]-x[:, 0:-2,0,:], -x[:,-1,0,:].reshape(self.rows,1,3)),axis=1)/2
            y2 = np.concatenate((x[0,:,1,:].reshape(1,self.cols,3),x[1:-1,:,1,:]-x[:-2,:,1,:],-x[-1,:,1,:].reshape(1,self.cols,3)),axis=0)/2
            y= -y1-y2
            return y
        elif self.canal ==1:
            # print('XDDDDDDD')
            y1 = np.concatenate((x[:, 0, 0].reshape(self.rows, 1), x[:, 1:-1, 0] - x[:, 0:-2, 0],
                                 -x[:, -1, 0].reshape(self.rows, 1)), axis=1) / 2.
            y2 = np.concatenate((x[0, :, 1].reshape(1, self.cols), x[1:-1, :, 1] - x[:-2, :, 1],
                                 -x[-1, :, 1].reshape(1, self.cols)), axis=0) / 2.
            y = -y1-y2
            # dh = np.array([[-1,1]])/2.
            # dv = np.array([[-1],[1]])/2.
            # otf_dh = psf2otf(dh, (self.rows,self.cols)).astype('complex128')
            # temp_dh = np.conj(otf_dh) * fftpack.fft2(x[:,:,0], axes=(0, 1))
            # temp_dh = fftpack.ifft2(temp_dh, axes=(0, 1))
            # temp_dh = np.real(temp_dh)


            # otf_dv = psf2otf(dv, (self.rows,self.cols)).astype('complex128')
            # temp_dv = np.conj(otf_dv) * fftpack.fft2(x[:,:,1], axes=(0, 1))
            # temp_dv = fftpack.ifft2(temp_dv, axes=(0, 1))
            # temp_dv = np.real(temp_dv)

            # y = temp_dh+temp_dv


            return y

    def L_nabla_f(self):
        xn = np.random.rand(*self.image.shape)
        rhon = 1+1e-2
        rhok = 1
        k=0
        xn1 = xn
        while abs(rhok - rhon)/abs(rhok) >= 1e-5:
            xn = xn1/np.linalg.norm(xn1,'fro')
            xn1 = 2*self.Dadjoint(self.optD(xn))
            rhon = rhok
            k+=1
            rhok = np.linalg.norm(xn1,'fro')
        return 1.01*np.sqrt(rhok)+1e-10

    def norm_ck_dk_opt(self,method,e):
        iter=0
        if (method == 'SL-PAM'):
#             xn = np.random.rand(*self.image.shape)
#             rhon = 1+1e-2
#             rhok = 1
#             k=0
#             xn1 = xn
            # while abs(rhok - rhon)/abs(rhok) >= 1e-5 and iter<5000:
            #     xn = xn1/np.linalg.norm(xn1,'fro')
            #     xn1 = self.S_du(xn,e)
            #     rhon = rhok
            #     k+=1
            #     rhok = np.linalg.norm(xn1,'fro')
            #     iter+=1
            rhok = 4
            return 1.01*self.beta*np.sqrt(rhok)+1e-8
        if (method == 'PALM'):
#             xn = np.random.rand(*self.image.shape)
#             rhon = 1 +1e-2
#             rhok_u = 1
#             rhok_e = 1
#             k = 0
#             xn1 = xn
            # while abs(rhok_u - rhon) / abs(rhok_u) >= 1e-5 and iter<5000:
            #     xn = xn1 / np.linalg.norm(xn1, 'fro')
            #     xn1 = self.S_du(xn, e)
            #     rhon = rhok_u
            #     k += 1
            #     rhok_u = np.linalg.norm(xn1, 'fro')
            #     iter+=1
            rhok_u=4

            # xn = np.random.rand(*self.en_PALM.shape)
            # xn = np.ones_like(self.en_PALM)
            # xn1 = xn
            iter=0
            # print(rhok_e)
#             while abs(rhok_e - rhon) / abs(rhok_e) >= 1e-5:# and iter<5000:
#                 xn[:,:,0] = xn1[:,:,0] / (np.linalg.norm(xn1[:,:,0], 'fro')+1e-10)
#                 xn[:, :,1] = xn1[:, :, 1] / (np.linalg.norm(xn1[:, :, 1], 'fro')+1e-10)
#                 xn1 = -2*(1-xn)*self.S_D(self.un_PALM)
#                 rhon = rhok_e
#                 k += 1
                # temp1 = np.linalg.norm(xn1[:,:,0], 'fro')
                # temp2 = np.linalg.norm(xn1[:, :, 1], 'fro')
                # rhok_e= np.sqrt(temp1**2+temp2**2)
#                 rhok_e= np.sqrt(np.sum(xn1**2))
                # iter+=!   
            rhok_e=4 

            return 1.01 *2* self.beta * np.sqrt(rhok_u)+1e-8,1.01*2*self.beta*np.sqrt(rhok_e)+1e-8
    def norm_ck_dk(self, method):
        #Calculate ck first, then dk
        bk = np.random.rand(self.rowscols * 2)
        
        if (method == 'SL-PAM'):
            e = ((1 - self.en_SLPAM)**2).ravel('C')
            
            block = diags([e.tolist()], [0])*self.mat
            rohnk=1
            rohnn=0
            # while(np.abs(rohnk-rohnn)/rohnk)>=1e-4:
            while np.abs(rohnk - rohnn)/np.abs(rohnk) >= 1e-5:
                rohnn= rohnk
                product = (block.dot(bk))
                bk = product / np.linalg.norm(product, 2)
                rohnk = bk.T.dot(block.dot(bk)) / (bk.T.dot(bk))
            ck_SLPAM = 1.01*self.beta*rohnk*2
            
            return ck_SLPAM

        # bk = np.random.rand(self.cols*self.rows*2)
        if (method == 'PALM'):
            e = ((1 - self.en_PALM)**2).ravel('C')
       
            block = diags([e.tolist()], [0])*self.mat
            dk_PALM=0
            rohnn=0
            rohnk=1
            while np.abs((rohnk-rohnn)/rohnk)>1e-5:
            # while np.abs(rohnk - rohnn) > 1e-4:
                rohnn= rohnk
                product = (block.dot(bk))
                bk = product / np.linalg.norm(product, 2)
                rohnk = bk.T.dot(block.dot(bk)) / (bk.T.dot(bk))
            ck_PALM = 1.01  *2* self.beta * rohnk+1e-10

            bk = np.random.rand(2*self.cols)
            rohnn = 0
            rohnk = 1
            if (self.canal==1):
                im_ravel = self.un_PALM.ravel('C')
                temp = (self.optD0.dot(im_ravel))
                temp = temp.reshape((2*self.rows,self.cols))
                mat = temp.dot(temp.T)
                while np.abs((rohnk-rohnn)/rohnk)>1e-5:
                # while np.abs(rohnk - rohnn) > 1e-4:
                    rohnn= rohnk
                    product = mat.dot(bk)
                    bk = product / np.linalg.norm(product, 2)
                    rohnk = bk.T.dot(mat.dot(bk))
                dk_PALM = 1.01 * self.beta * rohnk+1e-10

            # elif(self.canal==3):
            #     d = np.array([1e-10])
            #     for k in range(3):
            #         lambda0=0
            #         Dh = temp[:, :, 0,k]
            #         Dv = temp[:, :, 1,k]
            #         mat = np.vstack((Dh, Dv))
            #         mat = mat.dot(mat.T)
            #         bk = np.random.random(self.rows * 2)
            #         for i in range(N):
            #             product = mat.dot(bk)
            #             bk = product / np.linalg.norm(product, 2)
            #             lambda0 = bk.dot(mat.dot(bk.T)) / (bk.dot(bk.T))
            #         np.append(d,[lambda0])
            #     dk_PALM = 1.01 * self.beta * np.max(d)

            return ck_PALM,dk_PALM

    def opt_H(self):
        diagonals = [(0.5 * np.ones(self.cols - 1)).tolist(), (-0.5 * np.ones(self.cols)).tolist(), [0.5]]
        block = diags(diagonals, [1, 0, -self.cols + 1])
        opt_H = scp.sparse.block_diag([block for _ in range(self.rows)])
        # print(opt_H.todense())
        return opt_H

    def opt_V(self):
        diagonals = [-0.5 * np.ones(self.rows * self.cols), 0.5 * np.ones((self.rows - 1) * self.cols), 0.5 * np.ones(self.cols)]
        opt_V = diags(diagonals, [0, self.cols, -(self.rows * self.cols) + self.cols])
        # print(opt_V.todense())
        return opt_V

    
    def optD1_create(self):
        D1 = hstack((self.V,-self.H))
        return D1

    def optD0_create(self):
        D = vstack((self.H,self.V))
        return D


    def KLD(self,u,z,sig):
        nn= 200
        log2u = np.log(u) / np.log(2)
        log2u[np.isnan(log2u)] = 0
        log2u[np.isinf(log2u)] = 0
        kld = (z>0)*( u>0 )* (-z*log2u+sig*u) + (z>0)*( u<0)*nn + (z==0)*( u>=0)*(sig*u) + (z==0)*(u<0)*nn

        kld = np.sum(kld)
        return kld

    def loop_PALM_eps_descent(self):
        self.Jn_PALM = 1e10 * np.ones(self.MaximumIteration)
        self.time_PALM =  0
        self.ck_PALM = np.ones(self.MaximumIteration+1)
        self.dk_PALM = np.ones(self.MaximumIteration+1)
        err = 1
        self.energy(self.un_PALM, self.en_PALM, self.image_degraded)
        self.Jn_PALM[0] = self.J
        J_previous = self.J

        # Main loop
        time_start_PALM = time.time()
        it=0
        # print(self.beta,self.lam)
        images=[]
        while self.eps>=self.eps_AT_min:# and time.time()- time_start_PALM<60:
            print("Epsilon: ",self.eps)
            iteration = 0
            time_start_local = time.time()
            err=1
            while (err >= self.stop_criterion)  and iteration <self.MaximumIteration:# //np.ceil(np.log2(self.eps/0.02)): #and np.abs((temp_err-err)/err)>= self.stop_criterion):and time.time()- time_start_local<self.time_limit
                if self.optD_type == 'OptD':
                    ck, dk = self.norm_ck_dk_opt(method='PALM',e=self.en_PALM)
                elif self.optD_type == 'Matrix':
                    ck, dk = self.norm_ck_dk(method='PALM')


                next_un_PALM = self.L_prox(self.un_PALM - (self.beta/ ck) * self.S_du(self.un_PALM, self.en_PALM), 1 / ck,
                                      self.image_degraded)
                next_en_PALM = self.R_prox(self.en_PALM - (self.beta/ dk)*self.S_de(self.un_PALM,self.en_PALM),self.lam/dk)
                
                self.energy(next_un_PALM, next_en_PALM, self.image_degraded)
                J_current = self.J
                self.Jn_PALM[iteration] = self.J
                err =abs(J_current - J_previous)/abs(J_current+1e-8)
                if np.isnan(err):
                    # print('Nan error err')
                    break
                else:
                    self.un_PALM = next_un_PALM
                    self.en_PALM = next_en_PALM
       
                iteration+= 1
                J_previous = J_current
                
            fig=plt.figure(1,figsize=(10,10))
            plt.axis('off')
            plt.imshow(self.un_PALM,'gray',vmin=0,vmax=1)
            draw_contour(self.en_PALM,name='',fig=fig)
            plt.show()
            
            plt.figure()
            plt.plot(self.Jn_PALM[:iteration])
            plt.show()

            self.eps = self.eps/1.5

            it+= iteration

        self.time_PALM = time.time()- time_start_PALM
        

        self.image_reconstructed_PALM = self.un_PALM
        # f = plt.figure()
        # plt.imshow(self.image_reconstructed_PALM,'gray')
        # draw_contour(self.en_PALM,'',fig=f)
        # plt.show()
        self.en_PALM= np.ones_like(self.en_PALM)*(self.en_PALM>0.5)
        return self.en_PALM, self.image_reconstructed_PALM,self.image_degraded,it,self.time_PALM

    def loop_SL_PAM_eps_descent(self):
        self.Jn_SLPAM = 1e10 * np.ones(self.MaximumIteration + 1)
        # self.ck_SLPAM = np.ones(self.MaximumIteration+1)
        self.error_image_SLPAM = 1e5*np.ones(self.MaximumIteration+1)
        self.time_SLPAM = 0
        err = 1
        self.energy(self.un_SLPAM, self.en_SLPAM, self.image_degraded)
        it = 0

        self.Jn_SLPAM[it] = self.J
        
        # Main loop
        time_start_SLPAM = time.time()

        while self.eps>=self.eps_AT_min:# and time.time()- time_start_SLPAM<60:
            # print('Epsilon: ',self.eps)
            iteration=0
            time_start_local = time.time()
            err=1
            k=1
            while (err >= self.stop_criterion and iteration < self.MaximumIteration  and time.time()- time_start_local<(k+4)*60):
                temp_en_SLPAM = self.en_SLPAM  
                if self.optD_type == 'Matrix':
                    ck = self.norm_ck_dk(method='SL-PAM')
                elif self.optD_type =='OptD':
                    ck = self.norm_ck_dk_opt(method='SL-PAM',e=self.en_SLPAM)
                
                # ck = 1.01*2*self.beta

                # self.ck_SLPAM[it] = ck

                next_un_SLPAM= self.L_prox(self.un_SLPAM - (self.beta / ck) * self.S_du(self.un_SLPAM, self.en_SLPAM), 1/ck,
                                      self.image_degraded)

                
                if(self.norm_type!='AT' and  self.norm_type != 'AT-fourier'):

                    over = self.beta * self.S_D(self.un_SLPAM) + self.dk_SLPAM / 2. * self.en_SLPAM
                    lower= self.beta * self.S_D(self.un_SLPAM) + self.dk_SLPAM / 2.

                    next_en_SLPAM = self.R_prox(over / lower, self.lam / (2 * lower))


                else:

                    next_en_SLPAM = self.en_SLPAM + 2*self.beta/self.dk_SLPAM*self.optD(next_un_SLPAM)**2
                    
                    e_ravel_0 = next_en_SLPAM[:,:,0].ravel('C')
                    e_ravel_1 = next_en_SLPAM[:,:,1].ravel('C')
                    e_ravel = np.hstack((e_ravel_0,e_ravel_1))
                    C1 = 1+(self.lam/(2*self.eps*self.dk_SLPAM))
                    C2 = (2*self.lam*self.eps/self.dk_SLPAM)

                    hat = (2*self.beta/self.dk_SLPAM*self.optD(next_un_SLPAM)**2)
                    hat0 = hat[:,:,0].ravel('C')
                    hat1 = hat[:,:,1].ravel('C')
                    hat_conca = np.hstack((hat0,hat1))

                    block1 = csc_matrix(diags([hat_conca.tolist()], [0]))
                    block2 = csc_matrix(identity(self.rowscols*2)*C1)
                    block3 = C2*csc_matrix(self.optD1.T.dot(self.optD1))




                    mat = block1+block2+block3

                    
                    inv_mat = scp.sparse.linalg.inv(mat)
                    
                    temp1 = inv_mat.dot(e_ravel)#.todense()
                    temp = spsolve(mat,e_ravel)
                    # print(temp.shape)
                    # print(block1.shape,time.time()-time_start_local)
                    # print(temp-temp1)
                    next_en_SLPAM[:,:,0] = temp[:self.rowscols].reshape(self.rows,self.cols)
                    next_en_SLPAM[:,:,1] = temp[self.rowscols:].reshape(self.rows,self.cols)
            


                self.energy(next_un_SLPAM, next_en_SLPAM, self.image_degraded)
                self.Jn_SLPAM[iteration + 1] = self.J
                
                err = abs(self.Jn_SLPAM[iteration + 1] - self.Jn_SLPAM[iteration])/abs(self.Jn_SLPAM[iteration])
                
                if np.isnan(err):
                    # print('Nan error err')
                    break
                else:
                    self.un_SLPAM = next_un_SLPAM
                    self.en_SLPAM = next_en_SLPAM
                # if iteration%50==0:
                #     print('Iteration ',iteration, 'Error: ',err)
                iteration+= 1
                it+=1
                k+=1


            # f=plt.figure(1)
            # plt.axis('off')
            # plt.imshow(self.un_SLPAM,'gray',vmin=0,vmax=1)
            # draw_contour(self.en_SLPAM,name='',fig=f)
            # plt.savefig('{}.png'.format(self.eps))
            # plt.show()

            # f=plt.figure(1)
            # plt.axis('off')
            # plt.imshow(self.un_PALM,'gray',vmin=0,vmax=1)
            # draw_contour(self.en_PALM,name='',fig=f)
            # # plt.savefig('out_AT_eps_descent/{}.png'.format(self.eps))

            self.eps = self.eps/2
            self.time_SLPAM += time.time()- time_start_local
        self.image_reconstructed = self.un_SLPAM
        self.en_SLPAM= np.ones_like(self.en_SLPAM)*(self.en_SLPAM>0.5)
        return self.en_SLPAM, self.image_reconstructed,self.image_degraded,it,self.time_SLPAM

    def loop_SL_PAM(self):
        self.Jn_SLPAM = 1e10 * np.ones(self.MaximumIteration + 1)
        self.ck_SLPAM = np.ones(self.MaximumIteration+1)
        self.error_image_SLPAM = 1e5*np.ones(self.MaximumIteration+1)
        self.time_table = np.zeros(self.MaximumIteration+1)
        self.time_SLPAM = 0
        
        self.norm_xk_xkplus1 = np.zeros(self.MaximumIteration+1) 
        self.norm_ek_ekplus1 = np.zeros(self.MaximumIteration+1)
        self.cx = np.zeros(self.MaximumIteration+1)
        self.ce = np.zeros(self.MaximumIteration+1)
        self.Lnablaf= self.L_nabla_f()


        err = 1.
        self.energy(self.un_SLPAM, self.en_SLPAM, self.image_degraded)
        it = 0



        self.Jn_SLPAM[it] = self.J
        # Progress bar
        # pbar = ProgressBar(maxval=self.MaximumIteration,widgets=[Bar('=', '[', ']'), ' ', Percentage()])
        
        # pbar.start()

        # printProgressBar(0, self.MaximumIteration, prefix = 'Progress:', suffix = 'Complete', length = 50)

        # Main loop
        time_start_SLPAM = time.time()#
        while (err > self.stop_criterion)  and (it < self.MaximumIteration): #and time.time()-time_start_SLPAM<self.time_limit
            
            if self.optD_type == 'Matrix':
                ck = self.norm_ck_dk(method='SL-PAM')
            elif self.optD_type =='OptD':
                ck = self.norm_ck_dk_opt(method='SL-PAM',e=self.en_SLPAM)
            
            # ck = 1.01*2*self.beta

            self.ck_SLPAM[it] = ck

            self.un_SLPAM = self.L_prox(self.un_SLPAM - (self.beta / ck) * self.S_du(self.un_SLPAM, self.en_SLPAM), 1/ck,
                                  self.image_degraded)
            

            if(self.norm_type=='l1' or  self.norm_type == 'l1q'):
                over = self.beta * self.S_D(self.un_SLPAM) + self.dk_SLPAM / 2. * self.en_SLPAM
                lower= self.beta * self.S_D(self.un_SLPAM) + self.dk_SLPAM / 2.

                self.en_SLPAM = self.R_prox(over / lower, self.lam / (2 * lower))

                # self.un_SLPAM = self.un_SLPAM
                # self.ce[it] = -np.sum(self.en_SLPAM)*self.Lnablaf

            elif (self.norm_type=='l1l2'):

                
                mu= 1
                nu=1
                Lips_grad_f= 2
                Lips_f=  2
                rho_h= 1
                rho_l= 1
                Lips_g= 2

                self.a = 0.5*self.dk_SLPAM
                self.b = 0.5*self.ck_SLPAM[it]- np.sum((1-self.en_SLPAM)**2)**2/(2*mu)
                self.mx = 0.5*self.ck_SLPAM[it]+0.5*rho_l-Lips_grad_f**2*mu/2-nu*Lips_f**2/2
                self.me = 0.5*(self.dk_SLPAM+rho_h)
                self.Me = 0.5*self.dk_SLPAM+0.5*self.dk_SLPAM*Lips_g**2
                self.Mx = 0.5*self.ck_SLPAM[it]
                gamma = 1/self.ck_SLPAM[it]
                tau   = 1/self.dk_SLPAM
                temp1 = 1/(1+gamma*rho_l-Lips_grad_f**2*mu*gamma-gamma*nu**Lips_f**2)
                temp2 = (1/tau+Lips_g**2/nu)/(1/tau+rho_h)
                self.mu_final = np.array([temp1,temp2])
                

                over = self.beta * self.S_D(self.un_SLPAM) + self.dk_SLPAM / 2. * self.en_SLPAM
                lower= self.beta * self.S_D(self.un_SLPAM) + self.dk_SLPAM / 2.

                self.en_SLPAM = self.R_prox(over / lower, 1 / (2 * lower))

            elif self.norm_type=='AT':

                if(self.epsilon_log_descent is True):
                    self.eps= self.tab_eps[it]

                self.en_SLPAM = self.en_SLPAM + 2*self.beta/self.dk_SLPAM*self.optD(self.un_SLPAM)**2
                
                e_ravel_0 = self.en_SLPAM[:,:,0].ravel('C')
                e_ravel_1 = self.en_SLPAM[:,:,1].ravel('C')
                e_ravel = np.hstack((e_ravel_0,e_ravel_1))
                C1 = 1+(self.lam/(2*self.eps*self.dk_SLPAM))
                C2 = (2*self.lam*self.eps/self.dk_SLPAM)

                hat = (2*self.beta/self.dk_SLPAM*self.optD(self.un_SLPAM)**2)
                hat0 = hat[:,:,0].ravel('C')
                hat1 = hat[:,:,1].ravel('C')
                hat_conca = np.hstack((hat0,hat1))

                block1 = csc_matrix(diags([hat_conca.tolist()], [0]))
                block2 = csc_matrix(identity(self.rowscols*2)*C1)
                block3 = C2*csc_matrix(self.optD1.T.dot(self.optD1))




                mat = block1+block2+block3
                
                inv_mat = scp.sparse.linalg.inv(mat)
                
                temp = inv_mat.dot(e_ravel)#.todense()
                # temp = spsolve(mat,).todense()
                # print(temp.shape)


                self.en_SLPAM[:,:,0] = temp[:self.rowscols].reshape(self.rows,self.cols)
                self.en_SLPAM[:,:,1] = temp[self.rowscols:].reshape(self.rows,self.cols)

                

           
            self.energy(self.un_SLPAM, self.en_SLPAM, self.image_degraded)
            self.Jn_SLPAM[it + 1] = self.J
            
            err = abs(self.Jn_SLPAM[it + 1] - self.Jn_SLPAM[it])/abs(self.Jn_SLPAM[it + 1])
            self.time_table[it]  = time.time()-time_start_SLPAM
            # pbar.update(it)
            it += 1
            # printProgressBar(it, self.MaximumIteration, prefix = 'Progress:', suffix = 'Complete', length = 50)
            # plt.figure(1,figsize=(10,20))
            # plt.imshow(self.un_SLPAM,'gray',vmin=0,vmax=1)
            # draw_contour(self.un_SLPAM,self.en_SLPAM,'',number_of_image=1)
            # plt.show()
        # plt.figure(figsize=(10,10))
        # plt.hist(np.ravel(self.en_SLPAM))
        # plt.show()
        self.it_SLPAM  = it
        # pbar.finish()
        # print(time.time()-time_start_SLPAM,err)
        self.error_curve = np.log(np.abs(self.Jn_SLPAM[1:it] - self.Jn_SLPAM[:it-1])/np.abs(self.Jn_SLPAM[1:it])+1e-16)
        self.image_reconstructed = self.un_SLPAM
        self.time_SLPAM = time.time()-time_start_SLPAM
        self.en_SLPAM= np.ones_like(self.en_SLPAM)*(self.en_SLPAM>0.5)
        return self.en_SLPAM, self.image_reconstructed,self.image_degraded,self.Jn_SLPAM[:it],self.error_curve[:it],self.ck_SLPAM[:it],self.time_SLPAM,self.time_table[:it]
    

    def loop_iSL_PAM(self):
        self.Jn_SLPAM = 1e10 * np.ones(self.MaximumIteration + 1)
        self.ck_SLPAM = np.ones(self.MaximumIteration+1)
        self.error_image_SLPAM = 1e5*np.ones(self.MaximumIteration+1)
        self.time_SLPAM = 0
        err = 1.
        self.energy(self.un_SLPAM, self.en_SLPAM, self.image_degraded)
        it = 0

        self.Jn_SLPAM[it] = self.J
        # Progress bar
        widgets = ['Processing: ', Percentage(), ' ', Bar(marker='-', left='[', right=']'),
                   ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets, maxval=self.MaximumIteration)
        pbar.start()
        # Main loop
        time_start_SLPAM = time.time()

        en_SLPAM_previous = self.en_SLPAM
        un_SLPAM_previous = self.un_SLPAM

        while (err > self.stop_criterion) and (it < self.MaximumIteration): 
            
            if self.optD_type == 'Matrix':
                ck = self.norm_ck_dk(method='SL-PAM')
            elif self.optD_type =='OptD':
                ck = self.norm_ck_dk_opt(method='SL-PAM',e=self.en_SLPAM)
            
            # ck = 1.01*2*self.beta

            self.ck_SLPAM[it] = ck


            if self.alphabeta is True:
                ab = (it)/(it+3)
                alpha_u, beta_u = it, it 
                alpha_e, beta_e = it, it
            else:
                alpha_u, beta_u = 0.5, 0.5 
                alpha_e, beta_e = 0.5, 0.5

            y1 = self.un_SLPAM + alpha_u*(self.un_SLPAM-un_SLPAM_previous)
            z1 = self.un_SLPAM + beta_u*(self.un_SLPAM-un_SLPAM_previous)

            self.un_SLPAM = self.L_prox(y1 - (self.beta / ck) * self.S_du(z1, en_SLPAM_previous), 1/ck,
                                  self.image_degraded)

            
            if(self.norm_type!='AT' and  self.norm_type != 'AT-fourier'):

                y2 = self.en_SLPAM + alpha_e*(self.en_SLPAM-en_SLPAM_previous)
                z2 = self.en_SLPAM + beta_e*(self.en_SLPAM-en_SLPAM_previous)

                over = self.beta * self.S_D(self.un_SLPAM) + self.dk_SLPAM / 2. * en_SLPAM_previous
                lower= self.beta * self.S_D(self.un_SLPAM) + self.dk_SLPAM / 2.

                
                self.en_SLPAM = self.R_prox(over / lower, self.lam / (2 * lower))


            else:

                if(self.epsilon_log_descent is True):
                    self.eps= self.tab_eps[it]

                

                self.en_SLPAM = self.en_SLPAM + 2*self.beta/self.dk_SLPAM*self.optD(self.un_SLPAM)**2
                
                
                e_ravel_0 = self.en_SLPAM[:,:,0].ravel('C')
                e_ravel_1 = self.en_SLPAM[:,:,1].ravel('C')
                e_ravel = np.hstack((e_ravel_0,e_ravel_1))
                C1 = 1+(self.lam/(2*self.eps*self.dk_SLPAM))
                C2 = (2*self.lam*self.eps/self.dk_SLPAM)

                hat = (2*self.beta/self.dk_SLPAM*self.optD(self.un_SLPAM)**2)
                hat0 = hat[:,:,0].ravel('C')
                hat1 = hat[:,:,1].ravel('C')
                hat_conca = np.hstack((hat0,hat1))

                block1 = csc_matrix(diags([hat_conca.tolist()], [0]))
                block2 = csc_matrix(identity(self.rowscols*2)*C1)
                block3 = C2*csc_matrix(self.optD1.T.dot(self.optD1))




                mat = block1+block2+block3
                
                inv_mat = scp.sparse.linalg.inv(mat)
                
                temp = inv_mat.dot(e_ravel)#.todense()
                # temp = spsolve(mat,).todense()
                # print(temp.shape)


                self.en_SLPAM[:,:,0] = temp[:self.rowscols].reshape(self.rows,self.cols)
                self.en_SLPAM[:,:,1] = temp[self.rowscols:].reshape(self.rows,self.cols)

            un_SLPAM_previous = self.un_SLPAM
            en_SLPAM_previous = self.en_SLPAM

  
            self.energy(self.un_SLPAM, self.en_SLPAM, self.image_degraded)
            self.Jn_SLPAM[it + 1] = self.J
            
            err = abs(self.Jn_SLPAM[it + 1] - self.Jn_SLPAM[it])/abs(self.Jn_SLPAM[it + 1])
            it += 1
            pbar.update(it)
        self.it_SLPAM  = it
        pbar.finish()

        self.error_curve = np.log(np.abs(self.Jn_SLPAM[1:it] - self.Jn_SLPAM[:it-1])/np.abs(self.Jn_SLPAM[1:it])+1e-16)
        self.image_reconstructed = self.un_SLPAM
        self.time_SLPAM= time.time()-time_start_SLPAM
        self.en_SLPAM= np.ones_like(self.en_SLPAM)*(self.en_SLPAM>0.5)
        return self.en_SLPAM, self.image_reconstructed,self.image_degraded,self.Jn_SLPAM[:it],self.error_curve[:it],self.ck_SLPAM,self.time_SLPAM

    def loop_PALM(self):
        self.Jn_PALM = 1e10 * np.ones(self.MaximumIteration + 1)
        self.time_PALM = 0
        self.ck_PALM = np.ones(self.MaximumIteration+1)
        self.dk_PALM = np.ones(self.MaximumIteration+1)
        self.time_table=np.zeros(self.MaximumIteration+1)
        it = 0
        err = 1.
        self.energy(self.un_PALM, self.en_PALM, self.image_degraded)
        self.Jn_PALM[it] = self.J
        # Progress bar
        # widgets = ['Processing: ', Percentage(), ' ', Bar(marker='-', left='[', right=']'),
        #            ' ', ETA(), ' ', FileTransferSpeed()]
        # pbar = ProgressBar(widgets=widgets, maxval=self.MaximumIteration)
        # pbar.start()
        # Main loop
        # plt.imshow(self.un_PALM,'gray',vmin=0,vmax=1)
        # plt.show()
        time_start_PALM = time.time()
        while (err > self.stop_criterion)  and (it < self.MaximumIteration):#and  time.time()-time_start_PALM<self.time_limit:
            # ..............


            if self.optD_type == 'OptD':
                ck, dk = self.norm_ck_dk_opt(method='PALM',e=self.en_PALM)

            elif self.optD_type == 'Matrix':
                ck, dk = self.norm_ck_dk(method='PALM')

            # ck=2*self.beta*1.01
            self.ck_PALM[it] = ck
            self.dk_PALM[it] = dk
            self.un_PALM = self.L_prox(self.un_PALM - (self.beta/ ck) * self.S_du(self.un_PALM, self.en_PALM), 1 / ck,
                                  self.image_degraded)

            if(self.epsilon_log_descent is True):
                    self.eps= self.tab_eps[it]
                    # print(self.eps)

            self.en_PALM = self.R_prox(self.en_PALM - (self.beta/ dk)*self.S_de(self.un_PALM,self.en_PALM),self.lam/dk)
            
            
            self.energy(self.un_PALM, self.en_PALM, self.image_degraded)

            self.Jn_PALM[it + 1] = self.J
    

            err = abs(self.Jn_PALM[it + 1] - self.Jn_PALM[it])/abs(self.Jn_PALM[it + 1])
            self.time_table[it] = time.time()- time_start_PALM
            it += 1
            # pbar.update(it)
        # plt.figure(figsize=(10,10))
        # plt.hist(np.ravel(self.en_PALM),bins=10)
        # plt.show()    
        self.it_PALM  = it
        # print(time.time()-time_start_PALM,err)
        # pbar.finish()
        self.error_curve_PALM = np.log(np.abs(self.Jn_PALM[1:it] - self.Jn_PALM[:it-1])/np.abs(self.Jn_PALM[1:it]))
        self.image_reconstructed_PALM = self.un_PALM
        self.time_PALM = time.time()- time_start_PALM
        self.en_PALM= np.ones_like(self.en_PALM)*(self.en_PALM>0.5)
        return self.en_PALM, self.image_reconstructed_PALM,self.image_degraded,self.Jn_PALM[:it],self.error_curve_PALM[:it],self.dk_PALM[:it],self.ck_PALM[:it],self.time_PALM,self.time_table[:it]
    def loop_iPALM(self):
        self.Jn_PALM = 1e10 * np.ones(self.MaximumIteration + 1)
        self.time_PALM =  0
        self.ck_PALM = np.ones(self.MaximumIteration+1)
        self.dk_PALM = np.ones(self.MaximumIteration+1)
        it = 0
        err = 1.
        self.energy(self.un_PALM, self.en_PALM, self.image_degraded)
        self.Jn_PALM[it] = self.J
        # Progress bar
        widgets = ['Processing: ', Percentage(), ' ', Bar(marker='-', left='[', right=']'),
                   ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets, maxval=self.MaximumIteration)
        pbar.start()
        # Main loop
        # plt.imshow(self.un_PALM,'gray',vmin=0,vmax=1)
        # plt.show()
        time_start_PALM = time.time()
        
        un_PALM_previous = self.un_PALM
        en_PALM_previous = self.en_PALM
        # epsi = 0.000
        # alpha_bar = (1-epsi)*0.5-0.01
        # beta_bar = 1

        while (err > self.stop_criterion) and (it < self.MaximumIteration):
            # ..............
            


            # if self.optD_type == 'OptD':
            #     ck, dk = self.norm_ck_dk_opt(method='PALM',e=self.en_PALM)
            # elif self.optD_type == 'Matrix':
            #     ck, dk = self.norm_ck_dk(method='PALM')

            xn = np.random.rand(*self.image.shape)
            rhon = 1 + 1e-2
            rhok_u = 1
            rhok_e = 1
            k = 0
            xn1 = xn
            while abs(rhok_u - rhon) / abs(rhok_u) >= 1e-5:
                xn = xn1 / np.linalg.norm(xn1, 'fro')
                xn1 = self.S_du(xn, self.en_PALM)
                rhon = rhok_u
                k += 1
                rhok_u = np.linalg.norm(xn1, 'fro')

            ck= 1.01 * self.beta * np.sqrt(rhok_u)+1e-10              

            # ck=2*self.beta*1.01
            self.ck_PALM[it] = ck
            


            if self.alphabeta is True:
                ab = (it)/(it+3)
                alpha_u, beta_u = ab,ab
                alpha_e, beta_e = ab,ab
            else:
                alpha_u, beta_u = 0.4, 0.4
                alpha_e, beta_e = 0.4, 0.4

            # delta_1 = (alpha_bar+beta_bar)*ck/(1-epsi-2*alpha_bar)
            

            # delta_1 = (alpha_u+beta_u)*ck/(1-epsi-2*alpha_u)
            # tau_1 = ((1+epsi)*delta_1+(1+beta_u)*ck)/(1-alpha_u)
            
            tau_1 = ((1+2*beta_u)*ck)/((1-2*alpha_u))
            


            # tau_1 =  ck
            # tau_1 = ((1+2*beta_u)*ck)/(2*(1-alpha_u))

            y1 = self.un_PALM + alpha_u*(self.un_PALM-un_PALM_previous)
            z1 = self.un_PALM + beta_u*(self.un_PALM-un_PALM_previous)


            un_PALM_previous = self.un_PALM
            

            self.un_PALM = self.L_prox(y1 - (self.beta/ tau_1) * self.S_du(z1, self.en_PALM), 1 / tau_1,
                           self.image_degraded)


            if(self.epsilon_log_descent is True):
                    self.eps= self.tab_eps[it]

            y2 = self.en_PALM + alpha_e*(self.en_PALM-en_PALM_previous)
            z2 = self.en_PALM + beta_e*(self.en_PALM-en_PALM_previous)

            # if self.optD_type == 'OptD':
            #     ck, dk = self.norm_ck_dk_opt(method='PALM',e=self.en_PALM)
            # elif self.optD_type == 'Matrix':
            #     ck, dk = self.norm_ck_dk(method='PALM')

            xn = np.random.rand(*self.en_PALM.shape)
            xn1 = xn
            while abs(rhok_e - rhon) / abs(rhok_e) >= 1e-5:
                xn[:,:,0] = xn1[:,:,0] / np.linalg.norm(xn1[:,:,0], 'fro')
                xn[:, :,1] = xn1[:, :, 1] / np.linalg.norm(xn1[:, :, 1], 'fro')
                xn1 = 2*xn*self.S_D(self.un_PALM)
                rhon = rhok_e
                k += 1
                temp1 = np.linalg.norm(xn1[:,:,0], 'fro')
                temp2 = np.linalg.norm(xn1[:, :, 1], 'fro')
                rhok_e= np.sqrt(temp1**2+temp2**2)+1e-10
            dk= 1.01*self.beta*np.sqrt(rhok_e)
            self.dk_PALM[it] = dk

            # delta_2 = (alpha_e+2*beta_e)*dk/((1-epsi-2*alpha_e)*2.)
            # tau_2 = ((1+epsi)*delta_2+(1+beta_e)*dk)/(2-alpha_e)
            
            # tau_2 = (1+2*beta_e)*dk/(2*(1-alpha_e))
            tau_2 =(1+2*beta_e)*dk/((1-2*alpha_e))


            # tau_2 = dk
            en_PALM_previous = self.en_PALM
            
            self.en_PALM = self.R_prox(y2 - (self.beta/ tau_2)*self.S_de(self.un_PALM,z2),self.lam/tau_2)
            

            self.energy(self.un_PALM, self.en_PALM, self.image_degraded)

            self.Jn_PALM[it + 1] = self.J
            

            err = abs(self.Jn_PALM[it + 1] - self.Jn_PALM[it])/abs(self.Jn_PALM[it + 1])
            it += 1
            pbar.update(it)

        self.it_PALM  = it
        pbar.finish()
        self.error_curve_PALM = np.log(np.abs(self.Jn_PALM[1:it] - self.Jn_PALM[:it-1])/np.abs(self.Jn_PALM[1:it]))
        self.image_reconstructed_PALM = self.un_PALM
        self.time_PALM = time.time()- time_start_PALM
        self.en_PALM= np.ones_like(self.en_PALM)*(self.en_PALM>0.5)
        return self.en_PALM, self.image_reconstructed_PALM,self.image_degraded,self.Jn_PALM[:it],self.error_curve_PALM[:it],self.dk_PALM,self.ck_PALM,self.time_PALM

    def members(self):
        all_members = self.__dict__.keys()
        return [(item, self.__dict__[item]) for item in all_members if not item.startswith("_")]

    def ShowImageAll(self, cmap='gray', show=False):
        if show == True:
            if (self.method != 'compare'):
                f = plt.figure(figsize=(self.rows // 2, self.cols // 2))
                f1 = f.add_subplot(331)
                f1.title.set_text('Original image')
                f1.imshow(self.image, cmap=cmap,vmin=0,vmax=1)
                f2 = f.add_subplot(332)
                f2.title.set_text('Degradation')
                f2.imshow(self.image_degraded, cmap=cmap,vmin=0,vmax=1)
                if (self.method == 'SL-PAM'):
                    print('Draw SL-PAM')
                    f3 = f.add_subplot(333)
                    f3.title.set_text('SL-PAM reconstructed image')
                    f3.imshow(self.image_reconstructed, cmap=cmap,vmin=0,vmax=1)
                    f4 = f.add_subplot(334)
                    f4.title.set_text('Energy function')
                    f4.plot(self.Jn_SLPAM[1:self.it_SLPAM], label='Energy function')
                    f4.legend()
                    f5 = f.add_subplot(335)
                    f5.title.set_text('Error curve')
                    f5.plot(self.error_curve[1:self.it_SLPAM ], label='Error curve')
                    f5.legend()
                    yv, xv = np.where(self.en_SLPAM[:, :, 1] > 0.5)
                    yh, xh = np.where(self.en_SLPAM[:, :, 0] > 0.5)
                    f6 = f.add_subplot(337)
                    f6.title.set_text('SL-PAM contour')
                    f6.set_xlim([0, self.cols])
                    f6.set_ylim([0, self.rows])
                    f6.axis('equal')
                    if(self.draw=='points'):
                        f6.scatter(xv, yv, s=1)
                        f6.scatter(xh, yh, s=1)
                    elif(self.draw=='line'):
                        for i in range(0,len(xv)):
                            f6.plot([xv[i]-0.5,xv[i]+0.5],[self.cols-(yv[i]+0.5),self.cols-(yv[i]+0.5)], 'r-')
                        for i in range(0, len(xh)):
                            f6.plot([xh[i] + 0.5, xh[i] + 0.5], [self.cols - (yh[i] - 0.5), self.cols - (yh[i] + 0.5)],
                                        'r-')
                    
                    f7 = f.add_subplot(336)
                    f7.title.set_text('ck evolution')
                    f7.plot(self.ck_SLPAM[:self.it_SLPAM ])

                    # plt.savefig('../out/out_all/[{}][beta={}][lambda={}][dk={}][{}-{}-{}][{}-{}][{}].jpg'.format(self.method,self.beta,self.lam,self.dk_SLPAM,self.blur_type, self.blur_size, self.blur_std,
                    #                                                    self.noise_type, self.noise_std, self.save_name))
                    plt.show()

                    # plt.imshow(self.image_reconstructed,cmap='gray',vmin=0,vmax=1)
                    # plt.savefig('../out/out_restauration/[reconstructed-SL-PAM][Iteration={}][beta={}][lambda={}][{}-{}-{}][{}-{}][{}].jpg'.format(self.it_SLPAM ,self.beta,self.lam,self.blur_type, self.blur_size, self.blur_std,
                    #                                                    self.noise_type, self.noise_std, self.save_name))
                    # plt.show()
                elif (self.method == 'PALM'):
                    f3 = f.add_subplot(333)
                    f3.imshow(self.image_reconstructed_PALM, cmap=cmap,vmin=0,vmax=1)
                    f4 = f.add_subplot(334)
                    f4.plot(self.Jn_PALM[1:self.it_PALM ], label='Energy function')
                    f4.legend()
                    f5 = f.add_subplot(335)
                    f5.plot(self.error_curve_PALM[1:self.it_PALM ], label='Error curve')
                    f5.legend()

                    yv, xv = np.where(self.en_PALM[:, :, 1] >= 0.5)
                    yh, xh = np.where(self.en_PALM[:, :, 0] >= 0.5)

                    f6 = f.add_subplot(337)
                    f6.title.set_text('PALM contour')
                    f6.set_xlim([0, self.cols])
                    f6.set_ylim([0, self.rows])
                    f6.axis('equal')
                    if (self.draw == 'points'):
                        f6.scatter(xv, yv, s=1)
                        f6.scatter(xh, yh, s=1)
                    elif (self.draw == 'line'):
                        for i in range(0, len(xv)):
                            f6.plot([xv[i] - 0.5, xv[i] + 0.5], [self.cols - (yv[i] + 0.5), self.cols - (yv[i] + 0.5)],
                                    'r-')
                        for i in range(0, len(xh)):
                            f6.plot([xh[i] + 0.5, xh[i] + 0.5], [self.cols - (yh[i] - 0.5), self.cols - (yh[i] + 0.5)],
                                    'r-')



                    f7 = f.add_subplot(336)
                    f7.title.set_text('ck evolution')
                    f7.plot(self.ck_PALM[:self.it_PALM ],label='ck')
                    
                    f7.legend()

                    f8 = f.add_subplot(338)
                    f8.title.set_text('dk evolution')
                    f8.plot(self.dk_PALM[:self.it_PALM],label='dk')
                    f8.legend()

                    # plt.savefig('../out/out_all/[{}][beta={}][lambda={}][{}-{}-{}][{}-{}][{}].jpg'.format(self.method,self.beta,self.lam,self.blur_type, self.blur_size, self.blur_std,
                    #                                                    self.noise_type, self.noise_std, self.save_name))
                    plt.show()
                    # plt.imshow(self.image_reconstructed_PALM,cmap='gray',vmin=0,vmax=1)
                    # plt.savefig('../out/out_restauration/[reconstructed-PALM][beta={}][lambda={}][Iteration={}][{}-{}-{}][{}-{}][{}].jpg'.format(self.beta,self.lam,self.it2,self.blur_type, self.blur_size, self.blur_std,
                    #                                                    self.noise_type, self.noise_std, self.save_name))
                    # plt.close()
            else:
                f = plt.figure(figsize=(self.rows // 2, self.cols // 2))
                f1 = f.add_subplot(341)
                f1.title.set_text('Original image')
                f1.imshow(self.image, cmap=cmap,vmin=0,vmax=1)
                f2 = f.add_subplot(342)
                f2.title.set_text('Degradation')
                f2.imshow(self.image_degraded, cmap=cmap,vmin=0,vmax=1)
                f3 = f.add_subplot(344)
                f3.title.set_text('SL-PALM reconstructed image',vmin=0,vmax=1)
                f3.imshow(self.image_reconstructed, cmap=cmap)
                f8 = f.add_subplot(345)
                f8.title.set_text('PALM reconstructed image',vmin=0,vmax=1)
                f8.imshow(self.image_reconstructed_PALM, cmap=cmap,vmin=0,vmax=1)
                f4 = f.add_subplot(343)
                f4.title.set_text('Energy function')
                f4.plot(self.Jn_SLPAM[1:self.it_SLPAM ], label='SL-PAM energy')
                f4.plot(self.Jn_PALM[1:self.it_PALM ], label='PALM')
                f4.legend()
                f5 = f.add_subplot(346)
                f5.title.set_text('Error curve')
                f5.plot(self.error_curve[1:self.it_SLPAM ], label='SL-PAM')
                f5.plot(self.error_curve_PALM[1:self.it_PALM ], label='PALM')
                f5.legend()
                yv, xv = np.where(self.en_SLPAM[:, :, 1] > 0.5)
                yh, xh = np.where(self.en_SLPAM[:, :, 0] > 0.5)
                f6 = f.add_subplot(347)
                f6.title.set_text('SL-PAM contour')
                f6.set_xlim([0, self.cols])
                f6.set_ylim([0, self.rows])
                f6.axis('equal')
                if (self.draw == 'points'):
                    f6.scatter(xv, yv, s=1)
                    f6.scatter(xh, yh, s=1)
                elif (self.draw == 'line'):
                    for i in range(0, len(xv)):
                        f6.plot([xv[i] - 0.5, xv[i] + 0.5], [self.cols - (yv[i] + 0.5), self.cols - (yv[i] + 0.5)],
                                'r-')
                    for i in range(0, len(xh)):
                        f6.plot([xh[i] + 0.5, xh[i] + 0.5], [self.cols - (yh[i] - 0.5), self.cols - (yh[i] + 0.5)],
                                'r-')

                yv, xv = np.where(self.en_PALM[:, :, 1] >0.5)
                yh, xh = np.where(self.en_PALM[:, :, 0] >0.5)

                f7 = f.add_subplot(348)
                f7.title.set_text('PALM contour')
                f7.set_xlim([0, self.cols])
                f7.set_ylim([0, self.rows])
                f7.axis('equal')
                if (self.draw == 'points'):
                    f7.scatter(xv, yv, s=1)
                    f7.scatter(xh, yh, s=1)
                elif (self.draw == 'line'):
                    for i in range(0, len(xv)):
                        f7.plot([xv[i] - 0.5, xv[i] + 0.5], [self.cols - (yv[i] + 0.5), self.cols - (yv[i] + 0.5)],
                                'r-')
                    for i in range(0, len(xh)):
                        f7.plot([xh[i] + 0.5, xh[i] + 0.5], [self.cols - (yh[i] - 0.5), self.cols - (yh[i] + 0.5)],
                                'r-')
    
                    f9 = f.add_subplot(349)
                    f9.title.set_text('ck evolution')
                    f9.plot(self.ck[:self.it_SLPAM -1],label='ck SL-PAM')
                    f9.plot(self.ck_PALM[:self.it_PALM -1],label='ck PALM')
                    f9.legend()

                    f10 = f.add_subplot(3,4,10)
                    f10.title.set_text('norm error image ')
                    f10.plot(self.error_image_PALM[:self.it_PALM],label='PALM error')
                    f10.plot(self.error_image_SLPAM[:self.it_SLPAM ],label='SL-PALM error')
                    f10.legend()

                    f11 = f.add_subplot(3,4,11)
                    f11.title.set_text('dk evolution')
                    f11.plot(self.dk_PALM[:self.it_PALM  - 1], label='dk PALM')
                    f11.legend()
                    

                plt.savefig('../out/out_all/[{}][beta={}][lambda={}][dk={}][Iteration={}][norm={}][{}-{}-{}][{}-{}][{}].jpg'.format(self.method,self.beta,self.lam,self.dk_SLPAM,self.MaximumIteration,self.norm_type,self.blur_type, self.blur_size, self.blur_std,
                                                                   self.noise_type, self.noise_std, self.save_name))
                # plt.close()
                plt.show()
                # plt.imshow(self.image_reconstructed,cmap='gray',vmin=0,vmax=1)
                # plt.savefig('../out/out_restauration/[reconstructed-SL-PAM][beta={}][lambda={}][Iteration={}][{}-{}-{}][{}-{}][{}].jpg'.format(self.beta,self.lam,self.it_SLPAM +self.itadd,self.blur_type, self.blur_size, self.blur_std,
                #                                                    self.noise_type, self.noise_std, self.save_name))
                # plt.show()

                # plt.imshow(self.image_reconstructed_PALM,cmap='gray',vmin=0,vmax=1)
                # plt.savefig('../out/out_restauration/[reconstructed-PALM][beta={}][lambda={}][Iteration={}][{}-{}-{}][{}-{}][{}].jpg'.format(self.beta,self.lam,self.it_PALM +self.itadd,self.blur_type, self.blur_size, self.blur_std,self.noise_type, self.noise_std, self.save_name))
                # plt.close()

    def process(self):
        if (self.noised_image_input is None):
            self.Addblur()
            self.AddNoise()
            im = Image.fromarray((self.normalization(self.image_degraded) * 255).astype(np.uint8))

            if (self.noise_type == 'Gaussian' and self.blur_type == 'Gaussian'):
                im.save('../noisy_image/[{}-{}-{}][{}-{}][{}].png'.format(self.blur_type, self.blur_size, self.blur_std,
                                                                       self.noise_type, self.noise_std,
                                                                       self.save_name))
            elif (self.noise_type == 'Gaussian' and self.blur_type == 'Uniform'):
                im.save('../noisy_image/[{}-{}-{}][{}-{}][{}].png'.format(self.blur_type, self.blur_size, self.blur_std,
                                                                       self.noise_type, self.noise_std,
                                                                       self.save_name))
            elif self.noise_type == 'Poisson' and self.blur_type == 'Gaussian':
                im.save('../noisy_image/[{}-{}-{}][{}-{}][{}].png'.format(self.blur_type, self.blur_size, self.blur_std,
                                                                       self.noise_type, self.noise_peak,
                                                                       self.save_name))
            elif self.noise_type == 'Poisson' and self.blur_type == 'Uniform':
                im.save('../noisy_image/[{}-{}-{}][{}-{}][{}].png'.format(self.blur_type, self.blur_size, self.blur_std,
                                                                       self.noise_type, self.noise_peak,
                                                                       self.save_name))
            elif self.noise_type == 'none' and self.blur_type == 'Gaussian':
                im.save('../noisy_image/[{}-{}-{}][{}][{}].png'.format(self.blur_type, self.blur_size, self.blur_std,
                                                                    self.noise_type,
                                                                    self.save_name))
            elif self.noise_type == 'none' and self.blur_type == 'Uniform':
                im.save('../noisy_image/[{}-{}-{}][{}][{}].png'.format(self.blur_type, self.blur_size, self.blur_std,
                                                                    self.noise_type,
                                                                    self.save_name))
            elif self.blur_type == 'none' and self.noise_type == 'Gaussian':
                im.save('../../noisy_image/[{}][{}-{}][{}].png'.format(self.blur_type + '-blurring', self.noise_type,
                                                                 self.noise_std,
                                                                 self.save_name))
            elif self.blur_type == 'none' and self.noise_type == 'Poisson':
                im.save('../noisy_image/[{}][{}][{}].png'.format(self.blur_type + '-blurring', self.noise_type,
                                                              self.noise_peak,
                                                              self.save_name))



        elif self.blur_type != 'none': #If there is a blur_kernel
            # print('Used available blurred image ')
            if np.max(self.noised_image_input)>4:
                self.image_degraded = self.noised_image_input / 255.
            else:
                self.image_degraded = self.noised_image_input
            if self.blur_type == 'Gaussian':
                # fs = (self.blur_size - 1) / 2.
                # x = np.arange(-fs, fs + 1)
                # y = np.arange(-fs, fs + 1)
                # X, Y = np.meshgrid(x, y)
                # bump = np.exp(-1 / (2 * self.blur_std ** 2) * (X ** 2 + Y ** 2))
                # kernel = bump / np.sum(bump)

                # kernel = matlab_style_gauss2D((self.blur_size,self.blur_size),self.blur_std)
                #
                # # plt.imshow(kernel,'gray',vmin=0,vmax=1)
                # # plt.show()
                # if self.canal == 3:
                #     self.A = psf2otf(kernel, (self.rows, self.cols)).astype('complex128')
                #     self.A = np.repeat(self.A[:, :, np.newaxis], self.canal, axis=2)
                # elif self.canal == 1:
                #     self.A = psf2otf(kernel, (self.rows, self.cols)).astype('complex128')
                self.A= self.A
                

            elif self.blur_type == 'Uniform':
                kernel = np.ones((self.blur_size, self.blur_size))
                kernel = kernel / np.sum(kernel)
                self.A = psf2otf(kernel, (self.rows, self.cols)).astype('complex128')
                if self.canal == 3:
                    self.A = np.repeat(self.A[:, :, np.newaxis], self.canal, axis=2)
        elif self.blur_type=='none':
            # print("Used available degraded image, none blurring")
            if np.max(self.noised_image_input)>4:
                # print('0-255 type')
                self.image_degraded = self.noised_image_input / 255.
            else:
                # print('0-1 type')
                self.image_degraded = self.noised_image_input

        if (self.method == 'SLPAM'):
            self.initialisation_u_e_SLPAM(type_contour=self.type_contour)
            self.en_SLPAM, self.image_reconstructed,self.image_degraded,self.Jn_SLPAM,self.error_curve,self.ck_SLPAM,self.time_SLPAM,time_table= self.loop_SL_PAM()
            slpam_PSNR= PSNR(self.image,self.image_reconstructed)
            
            if self.norm_type=='l1l2':
                return self.en_SLPAM, self.image_reconstructed,self.image_degraded,self.Jn_SLPAM,self.error_curve,self.ck_SLPAM,self.time_SLPAM,self.Lnablaf,self.ce,self.a,self.b,self.mx,self.me,self.Mx,self.Me,self.mu_final
            else:
                return self.en_SLPAM, self.image_reconstructed,self.image_degraded,self.Jn_SLPAM,self.error_curve,self.ck_SLPAM,self.time_SLPAM,time_table
            

        elif (self.method == 'PALM'):
            self.initialisation_u_e_PALM(type_contour=self.type_contour)
            contour_PALM, restored_image_PALM, degraded_image_PALM, energy_PALM, error_PALM,dk_PALM,ck_PALM,time_PALM,time_table = self.loop_PALM()
            palm_PSNR= PSNR(self.image, restored_image_PALM)
            return contour_PALM, restored_image_PALM, degraded_image_PALM, energy_PALM, error_PALM,dk_PALM,ck_PALM,palm_PSNR,time_PALM,time_table
        elif (self.method == 'compare'):
            self.initialisation_u_e_SLPAM(type_contour=self.type_contour)
            contour_SL_PAM, restored_image_SL_PAM, degraded_image_SL_PAM, energy_SL_PAM, error_SL_PAM,self.ck_SLPAM = self.loop_SL_PAM()

            self.initialisation_u_e_PALM(type_contour=self.type_contour)
            contour_PALM, restored_image_PALM, degraded_image_PALM, energy_PALM, error_PALM, dk_PALM,ck_PALM = self.loop_PALM()
            return contour_SL_PAM, restored_image_SL_PAM, degraded_image_SL_PAM, energy_SL_PAM, contour_PALM, restored_image_PALM, degraded_image_PALM, energy_PALM, error_PALM
        elif(self.method=='SLPAM-eps-descent'):
            self.initialisation_u_e_SLPAM(type_contour=self.type_contour)
            contour_SL_PAM, restored_image_SL_PAM, degraded_image_SL_PAM,it,time_SLPAM= self.loop_SL_PAM_eps_descent()
            slpam_PSNR= PSNR(self.image, restored_image_SL_PAM)
            return contour_SL_PAM, restored_image_SL_PAM, degraded_image_SL_PAM,it,time_SLPAM
        elif(self.method=='PALM-eps-descent'):
            self.initialisation_u_e_PALM(type_contour=self.type_contour)
            contour_PALM, restored_image_PALM, degraded_image_PALM,it,time_PALM = self.loop_PALM_eps_descent()
            palm_PSNR= PSNR(self.image, restored_image_PALM)
            return contour_PALM, restored_image_PALM, degraded_image_PALM,it,time_PALM

        elif (self.method=='iPALM'):
            self.initialisation_u_e_PALM(type_contour=self.type_contour)
            contour_iPALM, restored_image_iPALM, degraded_image_iPALM, energy_iPALM,error_iPALM, dk_iPALM, ck_iPALM, time_iPALM = self.loop_iPALM()
            ipalm_PSNR= PSNR(self.image, restored_image_iPALM)
            return contour_iPALM, restored_image_iPALM, degraded_image_iPALM, energy_iPALM, error_iPALM,dk_iPALM,ck_iPALM,ipalm_PSNR,time_iPALM


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
def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

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
def draw_contour(e, name,fig=None,color='r',threshold=0.5):
    r, c, channel = np.shape(e)
    yv, xv = np.where(e[:, :, 1] > threshold)
    yh, xh = np.where(e[:, :, 0] > threshold)
    if fig == None:
        fig = plt.figure()#,figsize=(c//8, r//8)
    # f1 = fig.add_subplot(111)
    # plt.imshow(u,'gray',vmin=0,vmax=1)
    # f1.axis('equal')
    # f1.title.set_text(name)

    for i in range(0, len(xv)):
        plt.plot([xv[i] - 0.5, xv[i] + 0.5], [(yv[i] + 0.5), (yv[i] + 0.5)], color+'-')
    for i in range(0, len(xh)):
        plt.plot([xh[i] + 0.5, xh[i] + 0.5], [(yh[i] - 0.5), (yh[i] + 0.5)],
                color+'-') 
def PSNR(I,Iref):
    temp=I.ravel()
    tempref=Iref.ravel()
    NbP=I.size
    EQM=np.sum((temp-tempref)**2)/NbP
    b=np.max(np.abs(tempref))**2
    return 10*np.log10(b/EQM)


def SNR(I,Iref):
    return 20*np.log10(np.linalg.norm(Iref)/np.linalg.norm(I-Iref))

# define Jaccard Similarity function
# def jaccard(contour1, contour2):
#     hor1 = contour1[:, :, 0]
#     ver1 = contour1[:, :, 1]
#     hor2 = contour2[:, :, 0]
#     ver2 = contour2[:, :, 1]

#     hor1_edges = np.where(hor1 >= 0.5)
#     ver1_edges = np.where(ver1 >= 0.5)
#     hor2_edges = np.where(hor2 >= 0.5)
#     ver2_edges = np.where(ver2 >= 0.5)

#     array_list_hor1 = (np.array(hor1_edges).T).tolist()
#     array_list_ver1 = (np.array(ver1_edges).T).tolist()
#     array_list_hor2 = (np.array(hor2_edges).T).tolist()
#     array_list_ver2 = (np.array(ver2_edges).T).tolist()

#     nt1 = map(tuple, array_list_hor1)
#     nt2 = map(tuple, array_list_hor2)
#     st1 = set(nt1)
#     st2 = set(nt2)
#     intersection = len(list(st1 & st2))
#     union = (len(st1) + len(st2)) - intersection
#     jaccard_hor = float(intersection) / union

#     nt1 = map(tuple, array_list_ver1)
#     nt2 = map(tuple, array_list_ver2)
#     st1 = set(nt1)
#     st2 = set(nt2)
#     intersection = len(list(st1 & st2))
#     union = (len(st1) + len(st2)) - intersection
#     jaccard_ver = float(intersection) / union

#     return (jaccard_hor + jaccard_ver) / 2

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
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        # h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h    
class CreateNorms():
    def __init__(self, eps):
        self.eps = eps

    def L0(self, x):
        return np.sum(x != 0)

    def L1(self, x):
        return np.sum(np.abs(x))

    def quadL1(self, x):
        return np.sum(np.maximum(np.abs(x), x * x / self.eps))

    def L2(self, x):
        temp = np.sqrt(np.sum(x ** 2))
        return temp

    def L2D(self, x):
        return np.sum(x ** 2)

    def AT(self, optD1x,x):
        return self.eps * self.L2(optD1x) ** 2 + (0.25 / self.eps) * self.L2(x) ** 2
    # def L2_dual(self,x):
    # return
class ProximityOperators():
    def __init__(self, eps):
        self.eps = eps

    def L0(self, x, tau):
        return x * (np.abs(x) > np.sqrt(2 * tau))

    def L1(self, x, tau):
        return x - np.maximum(np.minimum(x, tau), -tau)

    def quadL1(self, x, tau):
        return np.maximum(0, np.minimum(np.abs(x) - tau,
                                        np.maximum(self.eps, np.abs(x) / (tau / (self.eps / 2.) + 1)))) * np.sign(x)

    def L2D(self, x, tau):
        return x / (1 + 2 * tau)

    def KLD(self, x, cof, sigma, tau):
        (x - tau * sigma + np.sqrt(np.abs(x - tau * sigma) ** 2 + 4 * tau * cof)) / 2.

    def L2(self, x, tau, z):
        return (x + tau * z) / (1 + tau)

    def L2_restoration(self, x, tau, z, otfA):
        temp = (fftpack.fft2(x, axes=(0, 1)) + tau * np.conj(otfA) * fftpack.fft2(z, axes=(0, 1))) / (
                    tau * np.conj(otfA) * otfA + 1)
        temp = fftpack.ifft2(temp, axes=(0, 1))
        temp = np.real(temp)
        return temp

    def L1L2(self,e,gamma_1,gamma_2,tau):
        return (1/(2*tau*gamma_2+1))*(e - np.maximum(np.minimum(e, tau*gamma_1), -tau*gamma_1))

## Print iterations progress
"""def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
   
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
 
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
"""