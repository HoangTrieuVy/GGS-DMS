import numpy as np


def GradientHor(x):
    y=x-np.roll(x,1,axis=1)
    y[:,0]=0
    return y/2

def GradientVer(x):
    y=x-np.roll(x,1,axis=0)
    y[0,:]=0
    return y/2

def DivHor(x):
    N=len(x[0])
    y=x-np.roll(x,-1,axis=1)
    y[:,0]=-x[:,1]
    y[:,N-1]=x[:,N-1]
    return y/2
def DivVer(x):
    N=len(x)
    y=x-np.roll(x,-1,axis=0)
    y[0,:]=-x[1,:]
    y[N-1,:]=x[N-1,:]
    return y/2

def opL(x):
    y=[]
    y.append(GradientHor(x))
    y.append(GradientVer(x))
    return np.asarray(y)

def optAdjL(y):
    x=DivHor(y[0])+DivVer(y[1])
    return x/2


def ProjGradBouleInf(g,l):
    gh=g[0]
    gv=g[1]
    temp=g
    p0=gh-(gh-l)*(gh>l)-(gh+l)*(gh<-l)
    p1=gv-(gv-l)*(gv>l)-(gv+l)*(gv<-l)
    temp[0]=p0
    temp[1]=p1
    return temp

def prox_normL2(x,gamma,y):
    return (x+gamma*y)/(1+gamma)

def prox_L12(y,l):
    ny = np.sqrt(y[0]**2+y[1]**2)
    ind = np.where(ny>l)
    
    ph = np.zeros_like(y[0])
    pv = np.zeros_like(y[1])
    
    ph[ind] = (1-l/ny[ind])*y[0][ind]
    pv[ind] = (1-l/ny[ind])*y[1][ind]
    
    return [ph,pv]