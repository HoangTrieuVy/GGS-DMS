import numpy as np
from scipy import fftpack
import time
from tools_dms import *
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
            print("Image gray scale")
            self.rows, self.cols = noised_image_input.shape
            self.canal = 1
            if np.max(noised_image_input) > 100:
                print('Convert image to float [0,1] \n')
                self.noised_image_input = (np.copy(noised_image_input)) / 255.0
            else:
                print('Image is already in float [0,1] \n')
                self.noised_image_input = np.copy(noised_image_input)
            self.en_SLPAM = np.zeros((self.rows,self.cols,2))
            self.en_PALM = np.zeros((self.rows,self.cols,2))
            self.un_SLPAM = self.noised_image_input
            self.un_PALM = self.noised_image_input
        elif size == 3:
            print("Color image")
            self.rows, self.cols, self.canal = noised_image_input.shape
            if np.max(noised_image_input) > 100:
                print('Convert image to float [0,1] \n')
                self.noised_image_input = (np.copy(noised_image_input)) / 255.0
            else:
                print('Image is already in float [0,1] \n')
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
        elif self.norm_type == "AT" and (self.method=='SLPAM'):
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

    def norm_ck_dk_opt(self, method):
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
                ck, dk = self.norm_ck_dk_opt(method="PALM")
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
                    ck = self.norm_ck_dk_opt(method="SLPAM")
                elif self.optD_type == "OptD":
                    ck = self.norm_ck_dk_opt(method="SLPAM")

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
            ck, dk = self.norm_ck_dk_opt(method="PALM")

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
            ck = self.norm_ck_dk_opt(method="SLPAM")
            self.un_SLPAM = self.L_prox(self.un_SLPAM- (self.beta / ck) * self.S_du(self.un_SLPAM, self.en_SLPAM), 1 / ck,self.noised_image_input)
            # next_un_SLPAM = self.L_prox(self.un_SLPAM- (self.beta / ck) * self.S_du(self.un_SLPAM, self.en_SLPAM), 1 / ck,self.noised_image_input)
            if self.norm_type == "l1" or self.norm_type == "l1q":
                over  = self.beta * self.S_D(self.un_SLPAM) + self.dk_SLPAM_factor*ck/ 2.0 * self.en_SLPAM
                lower = self.beta * self.S_D(self.un_SLPAM) + self.dk_SLPAM_factor*ck / 2.0
                self.en_SLPAM = self.R_prox(over / lower, self.lam / (2 * lower))
                self.Jn_SLPAM += [self.energy(self.un_SLPAM, self.en_SLPAM, self.noised_image_input)]
                err = abs(self.Jn_SLPAM[it + 1] - self.Jn_SLPAM[it]) / abs(self.Jn_SLPAM[it])
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


class CreateNorms:
    # def __init__(self):
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
