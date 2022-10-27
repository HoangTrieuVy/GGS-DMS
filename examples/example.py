import sys
sys.path.insert(0, '../python_dms/lib/')
from dms import *
from tools_dms import *
from tools_trof import *
from PIL import Image
import scipy as scp

import scipy.io
import argparse
import os 

def parse():
	parser = argparse.ArgumentParser(description='Discret Mumford-Shah functionnal')

	parser.add_argument('--z',type=str,help='noisy image path',default=None)
	parser.add_argument('--x',type=str,help='original image path',default=None)
	parser.add_argument('--b',type=float,help='beta',default=10)
	parser.add_argument('--l',type=float,help='lambda',default=2e-2)
	parser.add_argument('--algo',type=str,help='PALM, SLPAM,PALM-eps-descent,SLPAM-eps-descent',default='SLPAM')
	parser.add_argument('--norm',type=str,help='l1, AT',default='l1')
	parser.add_argument('--eps',type=float,help='epsilon',default=0.02)
	parser.add_argument('--it',type=int,help='number of iteration',default=300)

	return parser.parse_args()

def run(args):
	print('Discret Mumford-Shah functionnal using '+args.algo+'-'+args.norm)
	filename,extension_file = os.path.splitext(args.z)
	
	if extension_file == '.mat':
		data = scipy.io.loadmat(args.z)
		x0    = data['f']
		z 	 = data['fNoisy']
		exact_contour = data['e_exacte']
		A = data['A_python']
	elif extension_file=='.jpg' or extension_file=='.png':
		z = np.array(Image.open(args.z).convert('L'))
		if args.x is None:
			x0= None
		else:
			x0= np.array(Image.open(args.x).convert('L'))
		A  = np.ones_like(z)


	
	method  = args.algo
	normtype= args.norm
	if method =='PALM' or method=='PALM-eps-descent':
		if normtype=='AT':
			normtype= 'AT-fourier' 
	mit=args.it


	test = DMS('', noise_type='Gaussian',blur_type='Gaussian',
                               beta=args.b, lamb=args.l, method=method,MaximumIteration=mit,
                               noised_image_input=z, norm_type=normtype,stop_criterion=1e-4, dkSLPAM=1e-4,
                               optD='OptD',eps=args.eps,time_limit=360000,A=A)

	out = test.process()
	plt.figure(figsize=(10,5))

	if x0 is not None:
		plt.subplot(131)
		plt.imshow(x0,cmap='gray')
		plt.axis('off')
		plt.title('Original')
		ax1=plt.subplot(132)
		plt.imshow(z,cmap='gray')
		plt.axis('off')
		plt.title('Noisy')
		ax1.text(0.5,-0.1, 'PSNR: '+ str(format(PSNR(z,x0), '.2f')), size=12, ha="center", 
         transform=ax1.transAxes)
		ax2=plt.subplot(133)
		plt.imshow(out[1],cmap='gray')
		draw_contour(out[0],'',fig=ax2)
		ax2.text(0.5,-0.1, 'PSNR: '+  str(format(PSNR(out[1],x0), '.2f')), size=12, ha="center", 
         transform=ax2.transAxes)
		plt.axis('off')
		plt.title(method+'-'+normtype+'-'+'Denoised')
		plt.show()
	else:
		ax1= plt.subplot(121)
		ax1.imshow(z,cmap='gray')
		plt.axis('off')
		plt.title('Noisy')
		ax2=plt.subplot(122)
		ax2.imshow(out[1],cmap='gray')
		draw_contour(out[0],'',fig=ax2)
		plt.axis('off')
		plt.title(method+'-'+normtype+'-'+'Denoised')
		plt.show()

if __name__ =='__main__':
	args = parse()
	run(args)