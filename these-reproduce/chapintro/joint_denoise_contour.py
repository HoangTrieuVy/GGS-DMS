import sys
sys.path.insert(0, '../python_dms/lib/')
from tools_dms import *
from dms import *
from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.io
import time 

np.random.seed(0)

x = np.array(Image.open('../testset/10081.jpg'))/255.
r,l,_ = np.shape(x)
delta= 0.03
# print(np.random.normal(0,1,x.shape).shape)
z = x+ delta*np.random.normal(0,1,x.shape)


## Without GGS
slpam_solver = DMS(
        norm_type="l1",
        edges="similar",
        beta=8,
        lamb=1e-2,
        eps=0.2,
        stop_criterion=1e-4,
        MaximumIteration=50,
        method="SLPAM",
        noised_image_input=z,
        optD="OptD",
        dk_SLPAM_factor=1e-4,
        eps_AT_min=0.02,
        A=np.ones((r,l)))
palm_solver = DMS(
        norm_type="l1",
        edges="similar",
        beta=8,
        lamb=1e-1,
        eps=0.2,
        stop_criterion=1e-4,
        MaximumIteration=500,
        method="PALM",
        noised_image_input=z,
        optD="OptD",
        dk_SLPAM_factor=1e-4,
        eps_AT_min=0.02,
        A=np.ones((r,l)))

time1 = time.time()
out_slpam = slpam_solver.process()
time_slpam = time.time()-time1
time2 = time.time()
out_palm = palm_solver.process()
time_palm= time.time()-time2

print("SLPAM-CT:",time_slpam)
print("PALM-CT:",time_palm)

plt.figure()
plt.loglog(out_slpam[2],label='SLPAM')
plt.loglog(out_palm[2],label='PALM')
plt.legend()
plt.figure()
plt.grid("on")
plt.loglog(out_slpam[3],label='SLPAM')
plt.loglog(out_palm[3],label='PALM')
plt.legend()
plt.figure()
plt.grid("on")
plt.loglog(out_slpam[4],label='SLPAM')
plt.loglog(out_palm[4],label='PALM')
plt.legend()

plt.figure()
plt.subplot(121)
plt.imshow(out_slpam[1])
plt.subplot(122)
plt.imshow(out_palm[1])
plt.axis('off')
plt.figure()
ax1=plt.subplot(121)
plt.imshow(np.ones_like(x))
draw_contour(out_slpam[0], '', fig=ax1, color="r", threshold=0.5)
plt.axis('off')
ax2=plt.subplot(122)
plt.imshow(np.ones_like(x))
draw_contour(out_palm[0], '', fig=ax2, color="r", threshold=0.5)
plt.axis('off')

plt.show()