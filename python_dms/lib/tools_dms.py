import numpy as np
from scipy import fftpack
import time
# from dms import *
import numpy as np
from scipy import fftpack
import time
from scipy.sparse import diags
import scipy as scp
from scipy.sparse import hstack
from scipy.sparse import vstack
from scipy.sparse import csc_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve,lgmres
import matplotlib.pyplot as plt
# import cupy as cp
# from cupyx.scipy.sparse.linalg import spsolve as spsolve_cuda
# from cupyx.scipy.sparse import csc_matrix as csc_matrix_cuda

class DMS:
    def __init__(
        self,
        norm_type="l1",
        edges="similar",
        beta=8,
        lamb=1e-2,
        eps=0.2,
        stop_criterion=1e-4,
        MaximumIteration=5000,
        method="SL-PAM",
        noised_image_input=None,
        optD="OptD",
        dk_SLPAM_factor=1e-4,
        eps_AT_min=0.02,
        A=None):

        self.optD_type = optD
        self.eps = eps
        self.eps_AT_min = eps_AT_min
        self.MaximumIteration = MaximumIteration
        self.stop_criterion = stop_criterion
        self.noised_image_input = noised_image_input
        self.method = method
        self.beta = beta
        self.lam = lamb
        self.norm_type = norm_type
        self.edges = edges


        # Image variable
        shape = np.shape(noised_image_input)
        size = np.size(shape)
        if size == 2:
            self.rows, self.cols = noised_image_input.shape
            self.canal = 1
            if np.max(noised_image_input) > 100:
                print('Convert image to float \n')
                self.noised_image_input = (np.copy(noised_image_input)) / 255.0
            else:
                # print('Image in float')
                self.noised_image_input = np.copy(noised_image_input)
            self.en_SLPAM = np.zeros((self.rows,self.cols,2))
            self.en_PALM = np.zeros((self.rows,self.cols,2))
            self.un_SLPAM = self.noised_image_input
            self.un_PALM = self.noised_image_input
        elif size == 3:
            self.rows, self.cols, self.canal = noised_image_input.shape
            if np.max(noised_image_input) > 100:
                self.noised_image_input = (np.copy(noised_image_input)) / 255.0
            else:
                self.noised_image_input = np.copy(noised_image_input)
            self.canal = 3
            self.un_SLPAM = self.noised_image_input
            self.un_PALM = self.noised_image_input
            if edges == 'similar': # all channels RGB have the same contour
                self.en_SLPAM = np.zeros((self.rows,self.cols,2))
                self.en_PALM = np.zeros((self.rows,self.cols,2))
            elif edges == 'distinct':
                self.en_SLPAM = np.zeros((self.rows,self.cols,2, self.canal))
                self.en_PALM = np.zeros((self.rows,self.cols,2, self.canal))

        self.rowscols = self.rows * self.cols        
        # SL-PAM parameters
        self.error_curve = None
        self.dk_SLPAM_factor = dk_SLPAM_factor
        self.Jn_SLPAM = []
        self.A = A

        self.norms = CreateNorms()
        self.prox = ProximityOperators(self.eps)

        # PALM parameter
        self.Jn_PALM = []

        # Operator
        self.H = self.opt_H()
        self.V = self.opt_V()
        self.optD0 = self.optD0_create()
        self.optD1 = self.optD1_create()
        self.mat = self.optD0.dot(self.optD0.T)

        kernel2 = np.array([[1], [-1]])
        kernel1 = np.array([[1, -1]])
        self.v = psf2otf(kernel2, (self.rows, self.cols))
        self.h = psf2otf(kernel1, (self.rows, self.cols))

    def Aadjoint_process(self, im):
        img_ft = fftpack.fft2(im, axes=(0, 1))
        img_ft2 = np.conj(self.A) * img_ft
        self.Aadjoint = np.real(fftpack.ifft2(img_ft2, axes=(0, 1)))
        self.Aadjoint = self.Aadjoint
        return self.Aadjoint

    def perimeter_estimation(self, norm_type, method, en_SLPAM, en_PALM):
        if norm_type == "l1":
            if method == "PALM":
                return np.sum(
                    np.abs(en_PALM[:, :, 0] / self.rows)
                    + np.abs(en_PALM[:, :, 1] / self.cols)
                )
            elif method == "SL-PAM":
                return np.sum(
                    np.abs(en_SLPAM[:, :, 0] / self.rows)
                    + np.abs(en_SLPAM[:, :, 1] / self.cols)
                )
        if norm_type == "AT" or norm_type == "AT-fourier":
            if method == "PALM":
                e = en_PALM
            elif method == "SL-PAM":
                e = en_SLPAM
            optD1 = self.optD1_create()
            e_ravel_0 = e[:, :, 0].ravel("C") / self.rows
            e_ravel_1 = e[:, :, 1].ravel("C") / self.cols
            e_ravel = np.hstack((e_ravel_0, e_ravel_1))
            optD1e = optD1.dot(e_ravel)
            return (
                self.eps * self.norms.L2(optD1e) ** 2
                + (0.25 / self.eps) * self.norms.L2(e_ravel) ** 2
            )

    def degrade_matrix(self, x):
        if self.canal==1:
            img_ft = fftpack.fft2(x, axes=(0, 1))
            img_ft2 = self.A * img_ft
            self.Aadjoint = np.real(fftpack.ifft2(img_ft2, axes=(0, 1)))
            return self.Aadjoint
        elif self.canal==3:
            output= np.zeros_like(x)
            for i in range(3):
                img_ft = fftpack.fft2(x[:,:,i], axes=(0, 1))
                img_ft2 = self.A * img_ft
                output[:,:,i]= np.real(fftpack.ifft2(img_ft2, axes=(0, 1)))
            return output

    def L_term(self, u, z):
        temp = self.degrade_matrix(u) - z
        return 0.5 * (self.norms.L2(temp)) ** 2.0

    def L_prox(self, x, tau, z):
        return self.prox.L2_restoration(x, tau, z, self.A.astype("complex128"),self.canal)

    def S_term(self, u, e):
        if self.canal == 3:
            if self.edges=='similar':
                return self.beta*np.sum((np.repeat(((1 - e))[:, :, :, np.newaxis], self.canal, axis=3)* self.optD(u)) ** 2)
            elif self.edges =='distinct':
                pass
        elif self.canal == 1:
            return self.beta*np.sum(((1 - e) * self.optD(u)) ** 2)

    def S_du(self, u, e):
        if self.canal == 3:
            if self.edges=='similar':
                return 2 *self.Dadjoint((np.repeat(((1 - e))[:, :, :, np.newaxis], self.canal, axis=3)** 2* self.optD(u)) )
        elif self.canal == 1:
            return 2 *self.Dadjoint(self.optD(u)*(1 - e) ** 2)

    def S_de(self, u, e):
        if self.canal == 3:
            if self.edges=='similar':
                 return -2 * (1 - e) *self.beta* self.S_D(u)
        elif self.canal == 1:
            return -2 * (1 - e) *self.beta* self.S_D(u)

    def S_D(self, u):
        if self.canal == 3:
            temp = np.sum(self.optD(u) ** 2, 3)
            return temp
        elif self.canal == 1:
            temp = self.optD(u) ** 2
            return temp

    # contour length penalization
    def R_term(self, e):

        if self.norm_type == "l0":
            return self.norms.L0(e)
        elif self.norm_type == "l1":
            return self.norms.L1(e)
        elif self.norm_type == "l1q":
            return self.norms.quadL1(e)
        elif self.norm_type == "AT" or self.norm_type == "AT-fourier":
            optD1 = self.optD1_create()
            e_ravel_0 = e[:, :, 0].ravel("C")
            e_ravel_1 = e[:, :, 1].ravel("C")
            e_ravel = np.hstack((e_ravel_0, e_ravel_1))
            optD1e = optD1.dot(e_ravel)
            return self.norms.AT(optD1e=optD1e, e=e, eps=self.eps)
        elif self.norm_type == "l1l2":
            return self.gamma_1 * self.norms.L1(e) + self.gamma_2 * self.norms.L2D(e)

    def R_prox(self, e, tau):
        if self.norm_type == "l0":
            return self.prox.L0(e, tau)
        elif self.norm_type == "l1":
            return self.prox.L1(e, tau)
        elif self.norm_type == "l1q":
            return self.prox.quadL1(e, tau)
        elif self.norm_type == "AT":
            e_ravel_0 = e[:, :, 0].ravel("C")
            e_ravel_1 = e[:, :, 1].ravel("C")
            e_ravel = np.hstack((e_ravel_0, e_ravel_1))
            # e_ravel = csr_matrix(e_ravel)
            C1 = np.eye(self.rowscols * 2) * (1 + (2 * self.lam / (4 * self.eps * self.lam / tau)))
            C2 = np.ones((self.rowscols * 2, self.rowscols * 2)) * ( 2 * self.lam * self.eps * self.lam / tau)
            mat = csc_matrix(identity(self.rowscols * 2) * C1) + csc_matrix(C2 * self.optD1.T.dot(self.optD1))
            temp = spsolve(mat,e_ravel) #.todense()
            e[:, :, 0] = temp[: self.rowscols].reshape(self.rows, self.cols)
            e[:, :, 1] = temp[self.rowscols :].reshape(self.rows, self.cols)
            return e
        elif self.norm_type == "AT" and (self.method=='PALM' or self.method=='PALM-eps-descent' ):
            # tau is lambda/dk
            C1 = 1 + (tau / (2 * self.eps))
            C2 = 2 * tau * self.eps

            F = 1 / (C2 * np.conj(self.v) * self.v + C1)
            G = F * C2 * np.conj(self.v) * self.h
            J = C2 * np.conj(self.h) * self.h + C1
            K = C2 * np.conj(self.h) * self.v * G
            L = C2 * np.conj(self.h) * self.v
            e[:, :, 0] = np.real(
                fftpack.ifft2(
                    (F + G / (J - K) * L) * fftpack.fft2(e[:, :, 0], axes=(0, 1))
                    + G / (J - K) * fftpack.fft2(e[:, :, 1], axes=(0, 1)),
                    axes=(0, 1),
                )
            )
            e[:, :, 1] = np.real(
                fftpack.ifft2(
                    1 / (J - K) * L * fftpack.fft2(e[:, :, 0], axes=(0, 1))
                    + 1 / (J - K) * fftpack.fft2(e[:, :, 1], axes=(0, 1)),
                    axes=(0, 1),
                )
            )
            return e

    def energy(self, u, e, z):
        return self.L_term(u, z)+ self.beta * self.S_term(u, e)+ self.lam * self.R_term(e)

    # Difference operator D
    def optD(self, x):

        if self.canal == 3:
            y = np.zeros((self.rows, self.cols, 2, self.canal))
            y[:, :, 0, :] = (np.concatenate((x[:, 1:, :] - x[:, 0:-1, :], np.zeros((self.rows, 1, self.canal))),axis=1,)/ 2.0 )
            y[:, :, 1, :] = (np.concatenate( (x[1:, :, :] - x[0:-1, :, :], np.zeros((1, self.cols, self.canal))),axis=0,)/ 2.0)
            return y

        elif self.canal == 1:
            y = np.zeros((self.rows, self.cols, 2))
            # # print(temp.shape)
            y[:, :, 0] = (np.concatenate((x[:, 1:] - x[:, 0:-1], np.zeros((self.rows, 1))), axis=1)/ 2.0)
            y[:, :, 1] = (np.concatenate((x[1:, :] - x[0:-1, :], np.zeros((1, self.cols))), axis=0 )/ 2.0)
            return y

    def Dadjoint(self, x):
        if self.canal == 3:
            y1 = (np.concatenate((x[:, 0, 0, :].reshape(self.rows, 1, 3),x[:, 1:-1, 0, :] - x[:, 0:-2, 0, :],-x[:, -1, 0, :].reshape(self.rows, 1, 3),),axis=1,)/ 2) 
            y2 = (np.concatenate((x[0, :, 1, :].reshape(1, self.cols, 3),x[1:-1, :, 1, :] - x[:-2, :, 1, :],-x[-1, :, 1, :].reshape(1, self.cols, 3),),axis=0,)/ 2)
            y = -y1 - y2
            return y
        elif self.canal == 1:
            # print('XDDDDDDD')
            y1 = (np.concatenate((x[:, 0, 0].reshape(self.rows, 1), x[:, 1:-1, 0] - x[:, 0:-2, 0],-x[:, -1, 0].reshape(self.rows, 1),),axis=1,)/ 2.0)
            y2 = ( np.concatenate(( x[0, :, 1].reshape(1, self.cols),x[1:-1, :, 1] - x[:-2, :, 1],-x[-1, :, 1].reshape(1, self.cols),),axis=0,)/ 2.0)
            y = -y1 - y2
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
        rhon = 1 + 1e-2
        rhok = 1
        k = 0
        xn1 = xn
        while abs(rhok - rhon) / abs(rhok) >= 1e-5:
            xn = xn1 / np.linalg.norm(xn1, "fro")
            xn1 = 2 * self.Dadjoint(self.optD(xn))
            rhon = rhok
            k += 1
            rhok = np.linalg.norm(xn1, "fro")
        return 1.01 * np.sqrt(rhok) + 1e-10

    def norm_ck_dk_opt(self, method, e):
        iter = 0
        if method == "SLPAM":
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
            return 1.01 * self.beta * np.sqrt(rhok) + 1e-8
        if method == "PALM":
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
            rhok_u = 4

            # xn = np.random.rand(*self.en_PALM.shape)
            # xn = np.ones_like(self.en_PALM)
            # xn1 = xn
            iter = 0
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
            rhok_e = 4

            return (
                1.01 *  self.beta * np.sqrt(rhok_u) + 1e-8,
                1.01 *  self.beta * np.sqrt(rhok_e) + 1e-8,
            )

    def norm_ck_dk(self, method):
        # Calculate ck first, then dk
        bk = np.random.rand(self.rowscols * 2)

        if method == "SLPAM":
            e = ((1 - self.en_SLPAM) ** 2).ravel("C")

            block = diags([e.tolist()], [0]) * self.mat
            rohnk = 1
            rohnn = 0
            # while(np.abs(rohnk-rohnn)/rohnk)>=1e-4:
            while np.abs(rohnk - rohnn) / np.abs(rohnk) >= 1e-5:
                rohnn = rohnk
                product = block.dot(bk)
                bk = product / np.linalg.norm(product, 2)
                rohnk = bk.T.dot(block.dot(bk)) / (bk.T.dot(bk))
            ck_SLPAM = 1.01 * self.beta * rohnk * 2

            return ck_SLPAM

        # bk = np.random.rand(self.cols*self.rows*2)
        if method == "PALM":
            e = ((1 - self.en_PALM) ** 2).ravel("C")

            block = diags([e.tolist()], [0]) * self.mat
            dk_PALM = 0
            rohnn = 0
            rohnk = 1
            while np.abs((rohnk - rohnn) / rohnk) > 1e-5:
                # while np.abs(rohnk - rohnn) > 1e-4:
                rohnn = rohnk
                product = block.dot(bk)
                bk = product / np.linalg.norm(product, 2)
                rohnk = bk.T.dot(block.dot(bk)) / (bk.T.dot(bk))
            ck_PALM = 1.01 * 2 * self.beta * rohnk + 1e-10

            bk = np.random.rand(2 * self.cols)
            rohnn = 0
            rohnk = 1
            if self.canal == 1:
                im_ravel = self.un_PALM.ravel("C")
                temp = self.optD0.dot(im_ravel)
                temp = temp.reshape((2 * self.rows, self.cols))
                mat = temp.dot(temp.T)
                while np.abs((rohnk - rohnn) / rohnk) > 1e-5:
                    # while np.abs(rohnk - rohnn) > 1e-4:
                    rohnn = rohnk
                    product = mat.dot(bk)
                    bk = product / np.linalg.norm(product, 2)
                    rohnk = bk.T.dot(mat.dot(bk))
                dk_PALM = 1.01 * self.beta * rohnk + 1e-10

            return ck_PALM, dk_PALM

    def opt_H(self):
        diagonals = [
            (0.5 * np.ones(self.cols - 1)).tolist(),
            (-0.5 * np.ones(self.cols)).tolist(),
            [0.5],
        ]
        block = diags(diagonals, [1, 0, -self.cols + 1])
        opt_H = scp.sparse.block_diag([block for _ in range(self.rows)])
        # print(opt_H.todense())
        return opt_H

    def opt_V(self):
        diagonals = [
            -0.5 * np.ones(self.rows * self.cols),
            0.5 * np.ones((self.rows - 1) * self.cols),
            0.5 * np.ones(self.cols),
        ]
        opt_V = diags(diagonals, [0, self.cols, -(self.rows * self.cols) + self.cols])
        return opt_V

    def optD1_create(self):
        D1 = hstack((self.V, -self.H))
        return D1

    def optD0_create(self):
        D = vstack((self.H, self.V))
        return D

    def loop_PALM_eps_descent(self):
        self.time_PALM = 0
        err = 1
        self.Jn_PALM += [self.energy(self.un_PALM, self.en_PALM, self.noised_image_input)]
        list_u = []
        list_e = []
        # Main loop
        while self.eps >= self.eps_AT_min: 
            print("Epsilon: ", self.eps)
            iteration = 0
            err = 1
            while (err >= self.stop_criterion) and iteration < self.MaximumIteration:   
                ck, dk = self.norm_ck_dk(method="PALM")
                next_un_PALM = self.L_prox(
                    self.un_PALM
                    - (self.beta / ck) * self.S_du(self.un_PALM, self.en_PALM),
                    1 / ck,
                    self.noised_image_input,
                )
                next_en_PALM = self.R_prox(
                    self.en_PALM
                    - (self.beta / dk) * self.S_de(self.un_PALM, self.en_PALM),
                    self.lam / dk,
                )

        
                self.Jn_PALM += [self.energy(next_un_PALM, next_en_PALM, self.noised_image_input)]
                err = abs(self.Jn_PALM[iteration+1] - self.Jn_PALM[iteration]) / abs(self.Jn_PALM[iteration] + 1e-8)
                if np.isnan(err):
                    # print('Nan error err')
                    break
                else:
                    self.un_PALM = next_un_PALM
                    self.en_PALM = next_en_PALM

                iteration += 1
            self.eps = self.eps / 1.5
            list_u + [self.un_SLPAM]
            list_e + [self.en_SLPAM]

        return self.en_PALM,self.un_PALM,self.Jn_PALM,list_u,list_e

    def loop_SL_PAM_eps_descent(self):
        err = 1
        self.Jn_SLPAM += [self.energy(self.un_SLPAM, self.en_SLPAM, self.noised_image_input)]
        list_u = []
        list_e = []
        # Main loop
        while self.eps >= self.eps_AT_min: 
            print(self.eps)
            it = 0
            err = 1
            while (err >= self.stop_criterion and it < self.MaximumIteration):
                if self.optD_type == "Matrix":
                    ck = self.norm_ck_dk(method="SLPAM")
                elif self.optD_type == "OptD":
                    ck = self.norm_ck_dk_opt(method="SLPAM", e=self.en_SLPAM)

                self.un_SLPAM = self.L_prox(self.un_SLPAM- (self.beta / ck) * self.S_du(self.un_SLPAM, self.en_SLPAM),1 / ck,self.noised_image_input)
               
                self.en_SLPAM = (self.en_SLPAM+ 2 * self.beta / (self.dk_SLPAM_factor) * self.optD(self.un_SLPAM) ** 2)
                e_ravel_0 = self.en_SLPAM[:, :, 0].ravel("C")
                e_ravel_1 = self.en_SLPAM[:, :, 1].ravel("C")
                e_ravel = np.hstack((e_ravel_0, e_ravel_1))
                C1 = 1 + (self.lam / (2 * self.eps *(self.dk_SLPAM_factor)))
                C2 = 2 * self.lam * self.eps / (self.dk_SLPAM_factor)
                hat = 2 * self.beta / (self.dk_SLPAM_factor) * self.optD(self.un_SLPAM) ** 2
                hat0 = hat[:, :, 0].ravel("C")
                hat1 = hat[:, :, 1].ravel("C")
                hat_conca = np.hstack((hat0, hat1))
                # block1 = csc_matrix_cuda(diags([hat_conca.tolist()], [0])) 
                # block2 = csc_matrix_cuda(identity(self.rowscols * 2) * C1) 
                # block3 = C2 * csc_matrix_cuda(self.optD1.T.dot(self.optD1)) 
                block1 = csc_matrix(diags([hat_conca.tolist()], [0])) 
                block2 = csc_matrix(identity(self.rowscols * 2) * C1) 
                block3 = C2 * csc_matrix(self.optD1.T.dot(self.optD1)) 
                mat = block1 + block2 + block3
                # temp = cp.asnumpy(spsolve_cuda(mat, cp.asarray(e_ravel)))# spsolve(mat, e_ravel)
                # temp= spsolve(mat, e_ravel)
                temp,info = lgmres(mat, e_ravel)
                self.en_SLPAM[:, :, 0] = temp[: self.rowscols].reshape(self.rows, self.cols)
                self.en_SLPAM[:, :, 1] = temp[self.rowscols :].reshape(self.rows, self.cols)
                self.Jn_SLPAM += [self.energy(self.un_SLPAM, self.en_SLPAM, self.noised_image_input)]
                err = abs(self.Jn_SLPAM[it + 1] - self.Jn_SLPAM[it]) / abs(self.Jn_SLPAM[it])
                # print(err)
                if np.isnan(err):
                    break
                it += 1
            list_u + [self.un_SLPAM]
            list_e + [self.en_SLPAM]
            self.eps = self.eps / 1.5
        return self.en_SLPAM,self.un_SLPAM,self.Jn_SLPAM,list_u,list_e

    def loop_PALM(self):
        it = 0
        err = 1.0
        self.Jn_PALM+= [self.energy(self.un_PALM, self.en_PALM, self.noised_image_input)]
 
        while (err > self.stop_criterion) and ( it < self.MaximumIteration ):
            ck, dk = self.norm_ck_dk_opt(method="PALM", e=self.en_PALM)

            self.un_PALM = self.L_prox(
                self.un_PALM - (self.beta / ck) * self.S_du(self.un_PALM, self.en_PALM),
                1 / ck,
                self.noised_image_input,
            )

            self.en_PALM = self.R_prox(
                self.en_PALM - (self.beta / dk) * self.S_de(self.un_PALM, self.en_PALM),
                self.lam / dk,
            )

            self.Jn_PALM += [ self.energy(self.un_PALM, self.en_PALM, self.noised_image_input)]
            err = abs(self.Jn_PALM[it + 1] - self.Jn_PALM[it]) / abs( self.Jn_PALM[it + 1] )
            it += 1
        return self.en_PALM, self.un_PALM, self.Jn_PALM

    def loop_SL_PAM(self):
        err = 1.0
        self.Jn_SLPAM += [self.energy(self.un_SLPAM, self.en_SLPAM, self.noised_image_input)]
        it = 0      

        # Main loop
        while (err > self.stop_criterion) and (it < self.MaximumIteration):  
            ck = self.norm_ck_dk_opt(method="SLPAM", e=self.en_SLPAM)
            self.un_SLPAM = self.L_prox(self.un_SLPAM- (self.beta / ck) * self.S_du(self.un_SLPAM, self.en_SLPAM), 1 / ck,self.noised_image_input)
            # next_un_SLPAM = self.L_prox(self.un_SLPAM- (self.beta / ck) * self.S_du(self.un_SLPAM, self.en_SLPAM), 1 / ck,self.noised_image_input)
            if self.norm_type == "l1" or self.norm_type == "l1q":
                over  = self.beta * self.S_D(self.un_SLPAM) + self.dk_SLPAM_factor*ck/ 2.0 * self.en_SLPAM
                lower = self.beta * self.S_D(self.un_SLPAM) + self.dk_SLPAM_factor*ck / 2.0
                self.en_SLPAM = self.R_prox(over / lower, self.lam / (2 * lower))
            elif self.norm_type == "AT":
                self.en_SLPAM = (self.en_SLPAM+ 2 * self.beta / (self.dk_SLPAM_factor) * self.optD(self.un_SLPAM) ** 2)
                e_ravel_0 = self.en_SLPAM[:, :, 0].ravel("C")
                e_ravel_1 = self.en_SLPAM[:, :, 1].ravel("C")
                e_ravel = np.hstack((e_ravel_0, e_ravel_1))
                C1 = 1 + (self.lam / (2 * self.eps *(self.dk_SLPAM_factor)))
                C2 = 2 * self.lam * self.eps / (self.dk_SLPAM_factor)
                hat = 2 * self.beta / (self.dk_SLPAM_factor) * self.optD(self.un_SLPAM) ** 2
                hat0 = hat[:, :, 0].ravel("C")
                hat1 = hat[:, :, 1].ravel("C")
                hat_conca = np.hstack((hat0, hat1))
                # block1 = csc_matrix_cuda(diags([hat_conca.tolist()], [0])) 
                # block2 = csc_matrix_cuda(identity(self.rowscols * 2) * C1) 
                # block3 = C2 * csc_matrix_cuda(self.optD1.T.dot(self.optD1)) 
                block1 = csc_matrix(diags([hat_conca.tolist()], [0])) 
                block2 = csc_matrix(identity(self.rowscols * 2) * C1) 
                block3 = C2 * csc_matrix(self.optD1.T.dot(self.optD1)) 
                mat = block1 + block2 + block3
                # temp = cp.asnumpy(spsolve_cuda(mat, cp.asarray(e_ravel)))# spsolve(mat, e_ravel)
                # temp = spsolve(mat, e_ravel)
                temp,info = lgmres(mat, e_ravel)
                self.en_SLPAM[:, :, 0] = temp[: self.rowscols].reshape(self.rows, self.cols)
                self.en_SLPAM[:, :, 1] = temp[self.rowscols :].reshape(self.rows, self.cols)
                self.Jn_SLPAM += [self.energy(self.un_SLPAM, self.en_SLPAM, self.noised_image_input)]
                err = abs(self.Jn_SLPAM[it + 1] - self.Jn_SLPAM[it]) / abs(self.Jn_SLPAM[it])
                print(err)
            it += 1
        return self.en_SLPAM,self.un_SLPAM, self.Jn_SLPAM


    def process(self):

        if self.method == "SLPAM":
            return self.loop_SL_PAM()
        elif self.method == "PALM":
            return self.loop_PALM()
        elif self.method == "SLPAM-eps-descent":
            return  self.loop_SL_PAM_eps_descent()
        elif self.method == "PALM-eps-descent":
            return self.loop_PALM_eps_descent()

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

def psf2otf(psf, output_shape=None):
    """
    Convert a point spread function to the optical transfer function.

    Parameters
    ----------
    psf : ndarray
        The point spread function of the imaging system.
    output_shape : tuple, optional
        The shape of the output OTF array. If not provided, the shape of the
        OTF will be the same as the shape of the input PSF.

    Returns
    -------
    otf : ndarray
        The optical transfer function of the imaging system.
    """
    if output_shape is None:
        output_shape = psf.shape

    # Pad the PSF to the desired output shape
    padded_psf = np.zeros(output_shape, dtype=np.complex64)
    padded_psf[:psf.shape[0], :psf.shape[1]] = psf

    # Perform the FFT on the padded PSF
    otf = np.fft.fftn(padded_psf)

    # Normalize the OTF
    otf /= np.abs(otf).max()

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

def draw_contour(e, name, fig=None, color="r", threshold=0.5):
    r, c, channel = np.shape(e)
    yv, xv = np.where(e[:, :, 1] > threshold)
    yh, xh = np.where(e[:, :, 0] > threshold)
    if fig == None:
        fig = plt.figure()  # ,figsize=(c//8, r//8)

    for i in range(0, len(xv)):
        plt.plot(
            [xv[i] - 0.5, xv[i] + 0.5], [(yv[i] + 0.5), (yv[i] + 0.5)], color + "-"
        )
    for i in range(0, len(xh)):
        plt.plot(
            [xh[i] + 0.5, xh[i] + 0.5], [(yh[i] - 0.5), (yh[i] + 0.5)], color + "-"
        )

def PSNR(I, Iref):
    temp = I.ravel()
    tempref = Iref.ravel()
    NbP = I.size
    EQM = np.sum((temp - tempref) ** 2) / NbP
    b = np.max(np.abs(tempref)) ** 2
    return 10 * np.log10(b / EQM)

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

def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    # h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

class CreateNorms:
    # def __init__(self):
    #     print()
    def L0(self, x):
        return np.sum(x != 0)

    def L1(self, x):
        return np.sum(np.abs(x))

    def quadL1(self, x):
        return np.sum(np.maximum(np.abs(x), x * x / self.eps))

    def L2(self, x):
        temp = np.sqrt(np.sum(x**2))
        return temp

    def L2D(self, x):
        return np.sum(x**2)

    def AT(self, optD1e, e, eps):
        return eps * self.L2(optD1e) ** 2 + (0.25 / eps) * self.L2(e) ** 2

class ProximityOperators:
    def __init__(self, eps):
        self.eps = eps

    def L0(self, x, tau):
        return x * (np.abs(x) > np.sqrt(2 * tau))

    def L1(self, x, tau):
        return x - np.maximum(np.minimum(x, tau), -tau)

    def quadL1(self, x, tau):
        return np.maximum(
            0,
            np.minimum(
                np.abs(x) - tau,
                np.maximum(self.eps, np.abs(x) / (tau / (self.eps / 2.0) + 1)),
            ),
        ) * np.sign(x)

    def L2D(self, x, tau):
        return x / (1 + 2 * tau)

    def KLD(self, x, cof, sigma, tau):
        (x - tau * sigma + np.sqrt(np.abs(x - tau * sigma) ** 2 + 4 * tau * cof)) / 2.0

    def L2(self, x, tau, z):
        return (x + tau * z) / (1 + tau)

    def L2_restoration(self, x, tau, z, otfA,canal):
        if canal== 1:
            temp = (fftpack.fft2(x, axes=(0, 1))+ tau * np.conj(otfA) * fftpack.fft2(z, axes=(0, 1))) / (tau * np.conj(otfA) * otfA + 1)
            temp = fftpack.ifft2(temp, axes=(0, 1))
            temp = np.real(temp)
            return temp
        elif canal==3:
            output= np.zeros_like(x)
            for i in range(3):
                temp = (fftpack.fft2(x[:,:,i], axes=(0, 1))+ tau * np.conj(otfA) * fftpack.fft2(z[:,:,i], axes=(0, 1))) / (tau * np.conj(otfA) * otfA + 1)
                temp = fftpack.ifft2(temp, axes=(0, 1))
                output[:,:,i] = np.real(temp)
            return output
    def L1L2(self, e, gamma_1, gamma_2, tau):
        return (1 / (2 * tau * gamma_2 + 1)) * (
            e - np.maximum(np.minimum(e, tau * gamma_1), -tau * gamma_1)
        )

def draw_dots_multiresolution(b,a,beta_axis,lambda_axis,name='PSNR'):
    if name == 'Jaccard':
        stemp = 10
        vm=np.min(a[0])
        vM= np.max(a[-1])
    elif name == 'PSNR':
        vm= np.min(a[0])
        stemp = 1
        vM = np.max(a[-1])
    elif name == 'SSIM':
        vm= np.min(a[0])
        vM=np.max(a[-1])
    elif name == 'CE':
        vm= np.min(a)
        vM=np.max(a)
    y_label_list= []
    for item in beta_axis:
        y_label_list+= ['1e{}'.format(item)]
    x_label_list= []
    for item in lambda_axis:
        x_label_list+= ['1e{}'.format(item)]
    cm = plt.cm.get_cmap('RdYlBu')
    scale = 10
    plt.rcParams.update({'font.size': scale *4})
    size = len(b)
    fig,ax = plt.subplots(figsize=(size*scale,size*scale))
    ax.set_xticks(np.arange(len(x_label_list)))
    ax.set_yticks(np.arange(len(y_label_list)))
    ax.set_xticklabels(x_label_list)
    ax.set_yticklabels(y_label_list[::-1])

    plt.gca().invert_xaxis()
    ax.set_xlim(-2,5.5)
    ax.set_ylim(5.5,-1)
    plt.grid('on',linewidth=scale/10)
    plt.axis('equal')

    step=1
    size=size*1800   
    x= 2
    y= 2
    
    tx =[]
    ty =[]
    tz =[]
    ts =[]
    for i in range(-2,3):
        for j in range(-2,3):
            tx += [x+i]
            ty += [y+j]
            tz += [a[0][y+j,x+i]]
            ts += [size]
            plt.text(x+i-0.2,y+j,"{:10.2f}".format(a[0][y+j,x+i]),color='r')
    plt.scatter(tx,ty,c=tz,s=ts,cmap='winter',vmin=vm,vmax=vM,marker='s')
    plt.rcParams.update({'font.size': scale*2})
    dot = b[0]
    size=size/3
    xop,yop = dot[1],dot[0]
    x = x- (2*step-xop*step)
    y = y- (2*step-yop*step)
    step =  step /2
    tx =[]
    ty =[]
    tz =[]
    ts =[]
    for i in range(-2,3):
        for j in range(-2,3):
            tx += [x+i*step]
            ty += [y+j*step]
            tz += [a[1][2+j,2+i]]
            ts += [size]

    plt.scatter(tx,ty,c=tz,s=ts,marker='s',cmap='winter',vmin=vm,vmax=vM)
    ctab = "pygkmcb"
    for k in range(1,len(b)-1):
        colors = ctab[k]
        dot = b[k]
        xop,yop = dot[1],dot[0]
        x = x- (2*step-xop*step)
        y = y- (2*step-yop*step)
        size=size/3
        step =  step /2
        tx =[]
        ty =[]
        tz =[]
        ts =[]
        for i in range(-2,3):
            for j in range(-2,3):
                tx += [x+i*step]
                ty += [y+j*step]
                tz += [a[k+1][2+j,2+i]]
                ts += [size]
        plt.rcParams.update({'font.size': scale/(k)})
        cax=plt.scatter(tx,ty,c=tz,s=ts,marker='s',cmap='winter',vmin=vm,vmax=vM)
    cb=fig.colorbar(cax,ticks=[vm ,vM],orientation='vertical')   
    cb.ax.tick_params(labelsize=scale*6)
    plt.show()
    
def draw_multiresolution(a1,b1,color='Greens',add_text=True,vm=0,vM=1):    
    temp = np.copy(a1[0])
    size = temp.shape[0]
    row_curr = 0
    col_curr = 0
    row_prev = 0
    col_prev = 0
    a_curr= None
    a_prev= None
    r = 0
    for r in range(len(b1)):
        a_curr = np.zeros((size,size))
        print(b1[r])

        if r==0:
            a_curr = np.copy(a1[r])
            fig,ax = plt.subplots(figsize=(size,size))
            plt.imshow(a_curr,color,vmin=vm,vmax=vM)
            if add_text == True:
                for i in range(size):
                    for j in range(size):
                        text = ax.text(j, i, format(a_curr[i,j], '.2f'),
                            ha="center", va="center", color="blue")

            row_optim,col_optim = b1[r]

            row_curr = row_optim
            col_curr = col_optim

            row_prev = row_curr
            col_prev = col_curr
            print(row_prev,col_prev,row_curr,col_curr)
            plt.plot([col_prev-1.5,col_prev-1.5],[row_prev+1.5,row_prev-1.5],'g-',linewidth=5)
            plt.plot([col_prev-1.5,col_prev+1.5],[row_prev+1.5,row_prev+1.5],'g-',linewidth=5)
            plt.plot([col_prev-1.5,col_prev+1.5],[row_prev-1.5,row_prev-1.5],'g-',linewidth=5)
            plt.plot([col_prev+1.5,col_prev+1.5],[row_prev-1.5,row_prev+1.5],'g-',linewidth=5)
            plt.colorbar()
            plt.show()
            a_prev = a_curr


        else:
            fig,ax = plt.subplots(figsize=(size,size))

            row_optim,col_optim = b1[r-1]
            row_curr = row_prev*2
            col_curr = col_prev*2

            row_optim = row_curr +b1[r][0] -2
            col_optim = col_curr +b1[r][1] -2

            print(row_prev,col_prev,row_curr,col_curr)

            for i in range(size//2):
                for j in range(size//2):
                    a_curr[i*2,j*2]     = a_prev[i,j]
                    a_curr[i*2+1,j*2]   = a_prev[i,j]
                    a_curr[i*2,j*2+1]   = a_prev[i,j]
                    a_curr[i*2+1,j*2+1] = a_prev[i,j]

            a_curr[row_curr-2:row_curr+3,col_curr-2:col_curr+3]= a1[r]  
            a_prev = a_curr

    #         plt.plot([col_curr,0],[row_curr,0],'k-',linewidth=5)
            plt.plot([col_curr-2.5,col_curr+2.5],[row_curr-2.5,row_curr-2.5],'k-',linewidth=5)
            plt.plot([col_curr-2.5,col_curr+2.5],[row_curr+2.5,row_curr+2.5],'k-',linewidth=5)
            plt.plot([col_curr-2.5,col_curr-2.5],[row_curr-2.5,row_curr+2.5],'k-',linewidth=5)
            plt.plot([col_curr+2.5,col_curr+2.5],[row_curr-2.5,row_curr+2.5],'k-',linewidth=5)

    #         plt.plot([col_optim,5],[row_optim,3],'g-',linewidth=5)
            plt.plot([col_optim-1.5,col_optim+1.5],[row_optim-1.5,row_optim-1.5],'g-',linewidth=5)
            plt.plot([col_optim-1.5,col_optim+1.5],[row_optim+1.5,row_optim+1.5],'g-',linewidth=5)
            plt.plot([col_optim-1.5,col_optim-1.5],[row_optim-1.5,row_optim+1.5],'g-',linewidth=5)
            plt.plot([col_optim+1.5,col_optim+1.5],[row_optim-1.5,row_optim+1.5],'g-',linewidth=5)

            row_prev = row_curr 
            col_prev = col_curr


            if add_text == True:
                for i in range(size):
                    for j in range(size):
                        text = ax.text(j, i, format(a_curr[i,j], '.2f'),
                            ha="center", va="center", color="blue")
            plt.imshow(a_curr,color,vmin=vm,vmax=vM)
            plt.colorbar()
            plt.show()
        size += size

def golden_section_map(noised_im1,im1,contours_im1,bmax=5,bmin=-5,lmax=3,lmin=-6,scale_type='10',
                       grid_size=5,max_round=10,objective='Jaccard',maxiter=300,stop_crit=1e-4,method='SLPAM',norm_type='l1',eps=2.,eps_AT_min=0.02,A=None):
    out= None
    
    if scale_type  =='none':
        min_beta   = 1e-6
        max_beta   = 1e4
        min_lambda = 1e-7
        max_lambda = 1e3
        
        beta_axis   = np.linspace(min_beta,max_beta,grid_size)
        lambda_axis = np.linspace(min_lambda,max_lambda,grid_size)
    
        beta_list            = [beta_axis]
        lambda_list          = [lambda_axis]
    elif scale_type == '10':
        beta_axis   = np.linspace(bmax,bmin,grid_size) # Beta y-axis decreasing, because of python imshow
        lambda_axis = np.linspace(lmin,lmax,grid_size) # Keep lambda x-axis lambda increasing,
    
        beta_list            = [beta_axis]
        lambda_list          = [lambda_axis]
    
    print('Objective:',objective)
    temp = np.zeros((grid_size,grid_size))
    temp_ast = np.zeros((grid_size,grid_size))
    tab_PSNR_out   = np.zeros((max_round,grid_size,grid_size))
    tab_coord_max_PSNR_out   = []
    tab_PSNR_max = []
    
    tab_Jaccard_out = np.zeros((max_round,grid_size,grid_size))
    tab_coord_max_Jaccard_out  =  []
    tab_Jaccard_max =[]
    
    tab_CE_out = np.zeros((max_round,grid_size,grid_size))
    tab_coord_min_CE_out  =  []
    tab_CE_min =[]

    tab_SSIM_out   =  np.zeros((max_round,grid_size,grid_size))
    tab_coord_max_SSIM_out   = []
    tab_SSIM_max = []
    
    r=0
    time_start = time.time()


    while r < max_round:
        
        beta_axis_curr   = 10**beta_list[r]
        lambda_axis_curr = 10**lambda_list[r]
        if objective=='PSNR':   
            for i in range(grid_size):
                for j in range(grid_size):
                    
                    test = DMS(beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], method=method,MaximumIteration=maxiter,
                               noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, dk_SLPAM_factor=1e-4,
                               optD='OptD',eps=eps,A=A)

                    out = test.process()
                    temp[i,j]   = PSNR(out[1],im1)
                        
            # draw_table(temp,beta_axis_curr,lambda_axis_curr,color='Reds',vm=np.min(temp),vM=np.max(temp))
            tab_PSNR_out[r]    = temp
            tab_PSNR_max    += [np.max(tab_PSNR_out)]
            coord_max_PSNR_curr    = np.unravel_index(tab_PSNR_out[r].argmax(), tab_PSNR_out[r].shape)
            
            # Print out best PSNR for current round
            test = DMS(beta=beta_axis_curr[coord_max_PSNR_curr[0]], lamb=lambda_axis_curr[coord_max_PSNR_curr[1]], 
                        method=method,MaximumIteration=maxiter ,
                        noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                        dk_SLPAM_factor=1e-4,optD='OptD',eps=eps,A=A)

            out = test.process()
            # draw_result(restored_image= out[1],contour_detection= out[0])
            print('Round: ',r,' ',PSNR(out[1],im1))
            tab_coord_max_PSNR_out       += [coord_max_PSNR_curr]           
            beta_list += [np.linspace(beta_list[r][coord_max_PSNR_curr[0]]+(beta_list[r][-2]-beta_list[r][-1]),beta_list[r][coord_max_PSNR_curr[0]]-(beta_list[r][-2]-beta_list[r][-1]),grid_size)]
            lambda_list += [np.linspace(lambda_list[r][coord_max_PSNR_curr[1]]-(lambda_list[r][-1]-lambda_list[r][-2]),lambda_list[r][coord_max_PSNR_curr[1]]+(lambda_list[r][-1]-lambda_list[r][-2]),grid_size)]
            r+= 1

        elif objective=='Jaccard':
            for i in range(grid_size):
                for j in range(grid_size):
                    test = DMS(#beta=2.0548170999431815,lamb=0.002058421877614818,
                                beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], 
                                method=method,MaximumIteration=maxiter,
                               noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                               dk_SLPAM_factor=1e-4,optD='OptD',eps=eps,eps_AT_min=eps_AT_min,A=A)

                    out = test.process()
                    cont_thres = np.ones_like(out[0])*(out[0]>0.5)
                    temp[i,j]    = jaccard(cont_thres,contours_im1)

            tab_Jaccard_out[r]  = temp
            tab_Jaccard_max    += [np.max(temp)]
            coord_max_Jaccard_curr = np.unravel_index(tab_Jaccard_out[r].argmax(), tab_Jaccard_out[r].shape)

            # Print out best jaccard for current round
            test = DMS( beta=beta_axis_curr[coord_max_Jaccard_curr[0]], lamb=lambda_axis_curr[coord_max_Jaccard_curr[1]], 
                        method=method,MaximumIteration=maxiter ,
                        noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit,
                        dk_SLPAM_factor=1e-4,optD='OptD',eps=eps,eps_AT_min=eps_AT_min,A=A)

            out = test.process()
            cont_thres = np.ones_like(out[0])*(out[0]>0.5)

            print('Round: ',r, ' ',  jaccard(cont_thres,contours_im1),'beta:  ',beta_axis_curr[coord_max_Jaccard_curr[0]],', lam:  ',lambda_axis_curr[coord_max_Jaccard_curr[1]] )
            tab_coord_max_Jaccard_out       += [coord_max_Jaccard_curr]
            beta_list += [np.linspace(beta_list[r][coord_max_Jaccard_curr[0]]+(beta_list[r][-2]-beta_list[r][-1]),beta_list[r][coord_max_Jaccard_curr[0]]-(beta_list[r][-2]-beta_list[r][-1]),grid_size)]
            lambda_list += [np.linspace(lambda_list[r][coord_max_Jaccard_curr[1]]-(lambda_list[r][-1]-lambda_list[r][-2]),lambda_list[r][coord_max_Jaccard_curr[1]]+(lambda_list[r][-1]-lambda_list[r][-2]),grid_size)]
            r+= 1
        elif objective=='CE':
            for i in range(grid_size):
                for j in range(grid_size):

                    test = DMS(beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], method=method,MaximumIteration=maxiter,
                               noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                               dk_SLPAM_factor=1e-4,optD='OptD',eps=eps,eps_AT_min=eps_AT_min,A=A)

                    out = test.process()                    
                    cont_rec_torch= np.moveaxis(out[0],-1,0)
                    e_exacte_torch= np.moveaxis(contours_im1,-1,0)
                    cont_rec_torch = torch.tensor([cont_rec_torch],dtype=torch.float)
                    e_exacte_torch = torch.tensor([e_exacte_torch],dtype=torch.float)
                    temp[i,j]    = cross_entropy_loss2d(inputs=cont_rec_torch,targets=e_exacte_torch)
            tab_CE_out[r]  = temp
            tab_CE_min    += [np.min(temp)]
            coord_min_CE_curr = np.unravel_index(tab_CE_out[r].argmin(), tab_CE_out[r].shape)

            test = DMS( beta=beta_axis_curr[coord_min_CE_curr[0]], lamb=lambda_axis_curr[coord_min_CE_curr[1]], 
                        method=method,MaximumIteration=maxiter ,
                        noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit,
                        dk_SLPAM_factor=1e-4,optD='OptD',eps=eps,eps_AT_min=eps_AT_min,A=A)

            out = test.process()


            print('Round: ',r, 'iteration',len(out[3]),  cross_entropy_loss2d(inputs=cont_rec_torch,targets=e_exacte_torch))
            
            tab_coord_min_CE_out       += [coord_min_CE_curr]
            beta_list += [np.linspace(beta_list[r][coord_min_CE_curr[0]]+(beta_list[r][-2]-beta_list[r][-1]),beta_list[r][coord_min_CE_curr[0]]-(beta_list[r][-2]-beta_list[r][-1]),grid_size)]
            lambda_list += [np.linspace(lambda_list[r][coord_min_CE_curr[1]]-(lambda_list[r][-1]-lambda_list[r][-2]),lambda_list[r][coord_min_CE_curr[1]]+(lambda_list[r][-1]-lambda_list[r][-2]),grid_size)]
            r+= 1
                
        elif objective=='SSIM':
            for i in range(grid_size):
                for j in range(grid_size):
                    test = DMS(beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], method=method,MaximumIteration=maxiter,
                               noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                               dkSLPAM=1e-4,optD='OptD',eps=eps)

                    out = test.process()
                    temp[i,j]    = ssim(out[1],im1)
                       

            # draw_table(temp,beta_axis_curr,lambda_axis_curr,color='Purples',vm=0,vM=1)
            tab_SSIM_out[r]  = temp
            tab_SSIM_max    += [np.max(temp)]
            coord_max_SSIM_curr = np.unravel_index(tab_SSIM_out[r].argmax(), tab_SSIM_out[r].shape)
            # Print out best SSIM for current round
            test = DMS(noised_im1, '', noise_type='Gaussian',blur_type='none',
                               beta=beta_axis_curr[coord_max_SSIM_curr[0]], lamb=lambda_axis_curr[coord_max_SSIM_curr[1]], 
                               method=method,MaximumIteration=maxiter ,
                               noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                               dkSLPAM=1e-4,optD='OptD',eps=eps)

            out = test.process()
            print('Round: ',r)
            print('coord_max_SSIM_curr: ',coord_max_SSIM_curr,r'$\beta$: ',beta_axis_curr[coord_max_SSIM_curr[0]],'$\lambda$: ',lambda_axis_curr[coord_max_SSIM_curr[1]])
            
            if method != 'PALM-eps-descent' and method != 'SLPAM-eps-descent':
                print('SSIM: ', ssim(out[1],im1),'CT: ',out[-2],'seconds','   Iter:',len(out[3]))
            else:
                print('SSIM: ',  ssim(out[1],im1),'  CT: ',out[-2],'seconds','   Iter:',out[-3])
            
            tab_coord_max_SSIM_out       += [coord_max_SSIM_curr]
            beta_list += [np.linspace(beta_list[r][coord_max_SSIM_curr[0]]+(beta_list[r][-2]-beta_list[r][-1]),beta_list[r][coord_max_SSIM_curr[0]]-(beta_list[r][-2]-beta_list[r][-1]),grid_size)]
            lambda_list += [np.linspace(lambda_list[r][coord_max_SSIM_curr[1]]-(lambda_list[r][-1]-lambda_list[r][-2]),lambda_list[r][coord_max_SSIM_curr[1]]+(lambda_list[r][-1]-lambda_list[r][-2]),grid_size)]
            r+= 1
                
    print('\n\n\n')
    print('Meaningful r (optimum is in the middle)', r)
    time_elapsed = (time.time() - time_start)
    print('Total Computation time:', time_elapsed)
    if objective == 'PSNR':
        print('PSNR out')
        return tab_PSNR_out,tab_coord_max_PSNR_out,tab_PSNR_max,out[1],out[0]
    elif objective =='Jaccard':
        return tab_Jaccard_out,tab_coord_max_Jaccard_out,tab_Jaccard_max,out[1],out[0]
    elif objective =='SSIM':
        return tab_SSIM_out,tab_coord_max_SSIM_out,tab_SSIM_max,out[1],out[0]
    elif objective =='CE':
        return tab_CE_out,tab_coord_min_CE_out,tab_CE_min,out[1],out[0]

def grid_search(noised_im1,im1,contours_im1,bmax=5,bmin=-5,lmax=3,lmin=-6,scale_type='10',
                grid_size=5,max_round=10,objective='PSNR',maxiter=500,stop_crit=1e-4,method='SLPAM',
                norm_type='l1',eps=0.2,A=None):  
    
    if scale_type  =='none':
        min_beta   = 1e-6
        max_beta   = 1e4
        min_lambda = 1e-7
        max_lambda = 1e3
        
        beta_axis   = np.linspace(min_beta,max_beta,grid_size)
        lambda_axis = np.linspace(min_lambda,max_lambda,grid_size)
    
        beta_list            = [beta_axis]
        lambda_list          = [lambda_axis]
    elif scale_type == '10':
        beta_axis   = np.linspace(bmax,bmin,grid_size) # Beta y-axis decreasing, because of python imshow
        lambda_axis = np.linspace(lmin,lmax,grid_size) # Keep lambda x-axis lambda increasing,
    
        beta_list            = [beta_axis]
        lambda_list          = [lambda_axis]
    
    print('Objective:',objective)
    temp = np.zeros((grid_size,grid_size))
    tab_PSNR_out   = np.zeros((max_round,grid_size,grid_size))
    tab_coord_max_PSNR_out   = []
    tab_PSNR_max = []
    
    tab_Jaccard_out = np.zeros((max_round,grid_size,grid_size))
    tab_coord_max_Jaccard_out  =  []
    tab_Jaccard_max =[]

    tab_SSIM_out   =  np.zeros((max_round,grid_size,grid_size))
    tab_coord_max_SSIM_out   = []
    tab_SSIM_max = []
    
    
    beta_axis_curr   = 10**beta_list[0]
    lambda_axis_curr = 10**lambda_list[0]
    
    if objective=='PSNR':
        for i in range(grid_size):
            for j in range(grid_size):
                test = DMS(beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], method=method,MaximumIteration=maxiter,
                           noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, dkSLPAM=1e-4,
                           optD='OptD',eps=eps,A=A)

                out = test.process()
                temp[i,j]    = PSNR(out[1],im1)
                
        draw_table(temp,beta_axis_curr,lambda_axis_curr,color='winter',vm=0,vM=28)
        tab_PSNR_out[0]    = temp
        tab_PSNR_max    += [np.max(tab_PSNR_out)]
        
        coord_max_PSNR_curr    = np.unravel_index(tab_PSNR_out[0].argmax(), tab_PSNR_out[0].shape)
        # Print out best PSNR for current round
        tab_coord_max_PSNR_out       += [coord_max_PSNR_curr]

        test = DMS(beta=beta_axis_curr[coord_max_PSNR_curr[0]], lamb=lambda_axis_curr[coord_max_PSNR_curr[1]], 
                method=method,MaximumIteration=maxiter ,
                noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                dkSLPAM=1e-4,optD='OptD',eps=eps,A=A)

        out = test.process()
        draw_result(restored_image= out[1],contour_detection= out[0])
        print('Round: ',0)
        print('coord_max_PSNR_curr: ',coord_max_PSNR_curr,'beta: ',beta_axis_curr[coord_max_PSNR_curr[0]],'lambda: ',lambda_axis_curr[coord_max_PSNR_curr[1]])
        if method != 'PALM-eps-descent' and method != 'SLPAM-eps-descent':
            print('PSNR: ', PSNR(out[1],im1),'  CT: ',out[-2],'seconds','   Iter:',len(out[3]))
        else:
            print('PSNR: ', PSNR(out[1],im1),'  CT: ',out[-2],'seconds','   Iter:',out[-3])
    elif objective=='Jaccard':
        for i in range(grid_size):
            for j in range(grid_size):
                test = DMS(beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], method=method,MaximumIteration=maxiter,
                           noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                           dkSLPAM=1e-4,optD='OptD',eps=eps,A=A)

                out = test.process()

                temp[i,j] = jaccard(out[0],contours_im1)
        draw_table(temp,beta_axis_curr,lambda_axis_curr,color='winter',vm=0,vM=1)
        tab_Jaccard_out[0]  = temp
        tab_Jaccard_max    += [np.max(temp)]
        coord_max_Jaccard_curr = np.unravel_index(tab_Jaccard_out[0].argmax(), tab_Jaccard_out[0].shape)
        # Print out best Jaccard for current round
        tab_coord_max_Jaccard_out       += [coord_max_Jaccard_curr]
        test = DMS(beta=beta_axis_curr[coord_max_Jaccard_curr[0]], lamb=lambda_axis_curr[coord_max_Jaccard_curr[1]], 
                    method=method,MaximumIteration=maxiter ,
                    noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit,
                    dkSLPAM=1e-4,optD='OptD',eps=eps,A=A)

        out = test.process()
        draw_result(restored_image= out[1],contour_detection= out[0],gth_contour=contours_im1)
        print('Round: ',0)
        print('coord_max_Jaccard_curr: ',coord_max_Jaccard_curr,r'$\beta$: ',beta_axis_curr[coord_max_Jaccard_curr[0]],'$\lambda$: ',lambda_axis_curr[coord_max_Jaccard_curr[1]])

        if method != 'PALM-eps-descent' and method != 'SLPAM-eps-descent':
            print('Jaccard: ', jaccard(out[0],contours_im1),'CT: ',out[-2],'seconds','   Iter:',len(out[3]))
        else:
            print('Jaccard: ', jaccard(out[0],contours_im1),'  CT: ',out[-2],'seconds','   Iter:',out[-3])
        

    elif objective=='SSIM':
        for i in range(grid_size):
            for j in range(grid_size):
                test = DMS(beta=beta_axis_curr[i], lamb=lambda_axis_curr[j], method=method,MaximumIteration=maxiter,
                           noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                           dkSLPAM=1e-4,optD='OptD',eps=eps,A=A)

                out = test.process()

                temp[i,j]    = ssim(out[1],im1)

        draw_table(temp,beta_axis_curr,lambda_axis_curr,color='Purples',vm=0,vM=1)
        tab_SSIM_out[0]  = temp
        tab_SSIM_max    += [np.max(temp)]
        coord_max_SSIM_curr = np.unravel_index(tab_SSIM_out[0].argmax(), tab_SSIM_out[0].shape)
        # Print out best SSIM for current round
        tab_coord_max_SSIM_out       += [coord_max_SSIM_curr]
        test = DMS(beta=beta_axis_curr[coord_max_SSIM_curr[0]], lamb=lambda_axis_curr[coord_max_SSIM_curr[1]], 
                method=method,MaximumIteration=maxiter ,
                noised_image_input=noised_im1, norm_type=norm_type,stop_criterion=stop_crit, 
                dkSLPAM=1e-4,optD='OptD',eps=eps,A=A)

        out = test.process()
        draw_result(restored_image= out[1],contour_detection= out[0])
        print('Round: ',0)
        print('coord_max_SSIM_curr: ',coord_max_SSIM_curr,r'$\beta$: ',beta_axis_curr[coord_max_SSIM_curr[0]],'$\lambda$: ',lambda_axis_curr[coord_max_SSIM_curr[1]])

        if method != 'PALM-eps-descent' and method != 'SLPAM-eps-descent':
            print('SSIM: ', ssim(out[1],im1),'CT: ',out[-2],'seconds','   Iter:',len(out[3]))
        else:
            print('SSIM: ',  ssim(out[1],im1),'  CT: ',out[-2],'seconds','   Iter:',out[-3])
    if objective == 'PSNR':
        print('PSNR out')
        return tab_PSNR_out,tab_coord_max_PSNR_out,tab_PSNR_max,out[1],out[0]
    elif objective =='Jaccard':
        return tab_Jaccard_out,tab_coord_max_Jaccard_out,tab_Jaccard_max,out[1],out[0]
    
    elif objective =='SSIM':
        return tab_SSIM_out,tab_coord_max_SSIM_out,tab_SSIM_max,out[1],out[0]

def draw_table(tab,beta_axis,lambda_axis,color='Reds',vm=0,vM=28):
    fig, ax= plt.subplots(figsize=(10,10))
        
    y_label_list= []
    for item in beta_axis:
        y_label_list+= ['{:.2e}'.format(item)]
    x_label_list= []
    for item in lambda_axis:
        x_label_list+= ['{:.2e}'.format(item)]

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_label_list)))
    ax.set_yticks(np.arange(len(y_label_list)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_label_list)
    ax.set_yticklabels(y_label_list)


    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    im= plt.imshow(tab,cmap=color,vmin=np.min(tab),vmax=vM)
    # im=plt.pcolormesh(tab, edgecolors='k', linewidth=2)
    ax.set_aspect('equal')

    
    for i in range(len(y_label_list)):
        for j in range(len(x_label_list)):
            text = ax.text(j, i, format(tab[i,j], '.2f'),
            ha="center", va="center", color="blue")

    plt.rcParams["font.size"] = "15"
    fig.colorbar(im,fraction=0.06, pad=0.04,ticks=[np.min(tab), np.max(tab)])
    plt.show()
    
def draw_result(restored_image,contour_detection,gth_contour=None):
    temp = plt.figure(figsize=(10,10))
    plt.imshow(restored_image,'gray')
    plt.axis('off')
    draw_contour(contour_detection,'',fig=temp)

def draw_table_v2(tab,cmap,name=None,with_colorbar=False,save_link=None):
    fig, ax = plt.subplots(figsize=(35,35))
    y_label_list= []
    for item in beta_range:
        y_label_list+= [str(format(item, '.1f')) if item >=1e-3 else str(format(item, '.1f'))]
    x_label_list= []
    for item in exp_lambda_range:
        x_label_list+= [str(format(item, '.6f')) if item >=1e-3 else str(format(item, '.6'))]

    row,col = np.shape(tab)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_label_list)))
    ax.set_yticks(np.arange(len(y_label_list)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_label_list[::-1])
    ax.set_yticklabels(y_label_list[::-1])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")


    
    #     im = ax.imshow(tab[:len(y_label_list),:],cmap=cmap)

    if name =='energy':
        im = ax.imshow(tab[::-1,::-1],cmap=cmap,vmin=0,vmax=15)
        for i in range(len(y_label_list)):
            for j in range(len(x_label_list)):
                if np.abs(np.floor(tab[::-1,::-1][i,j]))<1:
                    text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.3f'),
                ha="center", va="center", color="black" if tab[::-1,::-1][i,j] < np.max(tab)/1.2 else "w")
                elif np.abs(np.floor(tab[::-1,::-1][i,j]))>10 and np.abs(np.floor(tab[::-1,::-1][i,j]))<100:
                    text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.2f'),
                ha="center", va="center", color="black" if tab[::-1,::-1][i,j] < np.max(tab)/1.2 else "w")
                else:
                    text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.2f'),
                ha="center", va="center", color="black" if tab[::-1,::-1][i,j] < np.max(tab)/1.2 else "w")
        
    elif name =='PSNR':
        im = ax.imshow(tab[::-1,::-1],cmap=cmap,vmin=5,vmax=15)
        for i in range(len(y_label_list)):
            for j in range(len(x_label_list)):
                if np.abs(np.floor(tab[::-1,::-1][i,j]))<1:
                    text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.3f'),
                ha="center", va="center", color="black" if tab[::-1,::-1][i,j] < np.max(tab)/1.2 else "w")
                elif np.abs(np.floor(tab[::-1,::-1][i,j]))>10 and np.abs(np.floor(tab[::-1,::-1][i,j]))<100:
                    text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.2f'),
                ha="center", va="center", color="black" if tab[::-1,::-1][i,j] < np.max(tab)/1.2 else "w")
                else:
                    text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.2f'),
                ha="center", va="center", color="black" if tab[::-1,::-1][i,j] < np.max(tab)/1.2 else "w")
        coord_max = np.unravel_index(tab.argmax(), tab.shape)
    #         plt.plot([col-1-1/2-coord_max[1],col-1-1/2-coord_max[1]+1],[row-1-1/2-coord_max[0],row-1-1/2-coord_max[0]],linewidth=2,c='green')
        draw_circle =plt.Circle((col-1-coord_max[1], row-1-coord_max[0]), 0.5,color='b', fill=True)
    #         draw_circle = plt.Circle((0.5, 0.5), 0.3,fill=False)

        ax.set_aspect(1)
        ax.add_artist(draw_circle)
    elif name=='time':
        im = ax.imshow(tab[::-1,::-1],cmap=cmap)
        for i in range(len(y_label_list)):
            for j in range(len(x_label_list)):
                if np.abs(np.floor(tab[::-1,::-1][i,j]))<1:
                    text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.3f'),
                ha="center", va="center", color="black" if tab[::-1,::-1][i,j] < np.max(tab)/1.2 else "w")
                elif np.abs(np.floor(tab[::-1,::-1][i,j]))>10 and np.abs(np.floor(tab[::-1,::-1][i,j]))<100:
                    text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.2f'),
                ha="center", va="center", color="black" if tab[::-1,::-1][i,j] < np.max(tab)/1.2 else "w")
                else :
                    text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.2f'),
                ha="center", va="center", color="black" if tab[::-1,::-1][i,j] < np.max(tab)/1.2 else "w")
    elif name=='difference':
        im = ax.imshow(tab[::-1,::-1],cmap=cmap,vmin=-0.3,vmax=0.3)
        for i in range(len(y_label_list)):
            for j in range(len(x_label_list)):
                text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.1f'),
            ha="center", va="center", color="black" if np.abs(tab[::-1,::-1][i,j]) < 1  else "w")
    elif name =='perimeter':
        im = ax.imshow(tab[::-1,::-1],cmap=cmap,vmin=0,vmax=3)
        for i in range(len(y_label_list)):
            for j in range(len(x_label_list)):
                text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.2f'),
            ha="center", va="center", color="black" if np.abs(tab[::-1,::-1][i,j]) < 5  else "w")
    else:    
        im = ax.imshow(tab[::-1,::-1],cmap=cmap,vmin=0,vmax=1)
        for i in range(len(y_label_list)):
            for j in range(len(x_label_list)):
                text = ax.text(j, i, format(tab[::-1,::-1][i,j], '.3f'),
            ha="center", va="center", color="black" if tab[::-1,::-1][i,j] < 0.75  else "w")
        
        coord_max = np.unravel_index(tab.argmax(), tab.shape)
    #         plt.plot([col-1-1/2-coord_max[1],col-1-1/2-coord_max[1]+1],[row-1-1/2-coord_max[0],row-1-1/2-coord_max[0]],linewidth=2,c='green')
        draw_circle =plt.Circle((col-1-coord_max[1], row-1-coord_max[0]), 0.5,color='b', fill=True)
    #         draw_circle = plt.Circle((0.5, 0.5), 0.3,fill=False)

        ax.set_aspect(1)
        ax.add_artist(draw_circle)
        
    plt.rcParams["font.size"] = "40"
    
    # Loop over data dimensions and create text annotations.
    if with_colorbar==True:        
        if name == 'Jaccard' or name=='SSIM':
            cbar = fig.colorbar(im,fraction=0.06, pad=0.04,ticks=[0, 1])
            cbar.ax.tick_params(labelsize=150)
        elif name=='difference':
            cbar = fig.colorbar(im,fraction=0.06, pad=0.04,ticks=[-0.3, 0.3])
            cbar.ax.set_yticklabels(['< -0.3', '> 0.3']) 
            cbar.ax.tick_params(labelsize=150)
        elif name=='perimeter':
            cbar = fig.colorbar(im,fraction=0.06, pad=0.04,ticks=[0, 3])
            cbar.ax.set_yticklabels(['0', '> 3']) 
            cbar.ax.tick_params(labelsize=150)
        else:
            cbar = fig.colorbar(im,fraction=0.06, pad=0.04,ticks=[1.7, 2.7])
            cbar.ax.set_yticklabels(['1.7', '2.7']) 
            cbar.ax.tick_params(labelsize=150)
    #     plt.axis('off')
    fig.tight_layout()
    if save_link != None:
        plt.savefig(save_link)
    plt.show()

