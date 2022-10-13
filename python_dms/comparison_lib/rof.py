import numpy as np
from ROF_tools import *
from progressbar import *


def PA_PDtv(I,chi,Niter = 500):
    ## Some fixed parameter 
    gamma = 0.99
    mu_g  = 1
    normD = np.sqrt(2)
    tau   = gamma/normD
    sig   = gamma/normD
    eps   = 1e-4
    
    
    
    ## Initializing variables
    x_n = np.zeros_like(I)
    v_n = opL(I)
    y_n = np.zeros_like(v_n)
    
    primal_crit = 1e10*np.ones(Niter)
    dual_crit = 1e10*np.ones(Niter)
    ## ALGORITHM ##
    k=0
    eps_crit= 1e10
    widgets = ['Processing: ', Percentage(), ' ', Bar(marker='-', left='[', right=']'),
                   ' ', ETA(), ' ', FileTransferSpeed()]
    
    pbar = ProgressBar(widgets=widgets, maxval=Niter)
    pbar.start()
        
    while (k<Niter-1 and eps_crit>=1e-4):
        
        # Save the dual variable
        v_n_previous = v_n
        
        # Update primal variable x
        x_n = prox_normL2(x_n-tau*optAdjL(v_n),tau,I)
        
        # Update dual variable v
        temp_0 = opL(x_n)
        temp_1 = v_n+sig*temp_0
        temp_2 = prox_L12(temp_1/sig,chi/sig)
        
        v_n[0] =  temp_1[0]-sig*temp_2[0]
        v_n[1] =  temp_1[1]-sig*temp_2[1]
        
        #Update of the descent steps
        theta = (1+2*mu_g*tau)**(-1/2);
        tau = theta*tau;
        sig=sig/theta;
        
        #Update dual auxiliary variable
        y_n[0] = v_n[0] + theta*(v_n[0] - v_n_previous[0])
        y_n[1] = v_n[1] + theta*(v_n[1] - v_n_previous[1])
        
        # Dual crit
        p = optAdjL(y_n)
        [p1,p2] = prox_L12(y_n,chi)
        dual_crit[k+1] = 0.5*np.sum(p**2)-chi*np.sum(p*I)+np.sum(np.sqrt(p1**2+p2**2))
        #table_energy[k] = 0.5*np.sum((DivHor(v_n))**2)+lamb*np.sum(np.abs(GradientHor(y+DivHor(v_n))))        
        
        # Primal crit
        [ph,pv] = opL(x_n)
        primal_crit[k+1] = 0.5*np.sum((I-x_n)**2)+chi*np.sum(np.sqrt(np.sum(ph**2+pv**2)))
        
        eps_crit = np.abs(primal_crit[k+1]-primal_crit[k]/primal_crit[k])
        k+=1
        pbar.update(k)
        
#         table_PSNR[k] = PSNR(y+Gradient(v_n),img1)
#         table_PSNR[k] = PSNR(x,img1)
#     x = y+DivHor(u_n)
    pbar.finish()
    
    return x_n,primal_crit,dual_crit
    
def trof(f,K):
    #Inputs: -f: denoised imag
    #        -K: number of segments
    #Outputs:
    #        -pc_res: piecewise constant image
    #        -seg_map: map of label
    
    Niter= 20
    tau = np.linspace(np.min(f),np.max(f),K)
    m0 = np.mean(f[np.where(tau[0]>=f)])
    ind = []
    m  = np.zeros(K-1)
    
    ## T-ROF ALGORITHM
    for i in range(Niter):
        for k in range(K-1):
            ind+= [np.where((f>tau[k])&(tau[k+1]>=f))]
            m[k] = np.mean(f[ind[k]])
            if k!=1:
                tau[k] = 0.5*(m[k-1]+m[k])
            else:
                tau[k] = 0.5*(m0+m[k])
            m0 = np.mean(f[np.where(tau[0]>=f)])
    seg_map = np.ones(np.shape(f))
    pc_res = m0*np.ones(np.shape(f))
    for k in range(K-1):
        seg_map[f>tau[k]] = k+1
        pc_res[f>tau[k]] = m[k]
    return pc_res,seg_map
def normalization(x):
    z = (x - np.min(x)) / (np.max(x) - np.min(x))
    return z
def optD_normalized(x):
    rows, cols = x.shape
    y = np.zeros((rows, cols, 2))
    # # print(temp.shape)
    y[:, :, 0] = np.concatenate((x[:, 1:] - x[:, 0:-1], np.zeros((rows, 1))),axis=1) / 2.
    y[:, :, 1] = np.concatenate((x[1:, :] - x[0:-1, :], np.zeros((1, cols))),axis=0) / 2.
    y= np.abs(y)
    y= y>0
    return y

def mix_PAPD_trof(f,chi,K):
    denoised_image,primal_crit,dual_crit = PA_PDtv(f, chi)
    pc_res,seg_map = trof(denoised_image,K)
    return denoised_image,pc_res,seg_map,primal_crit,dual_crit

