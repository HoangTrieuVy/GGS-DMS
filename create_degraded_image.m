clear all;
clc;
setPath;
%Parameters of degraded image
name_image= 'example1';
path= 'BSDS300_images/images/train/12003';
name_image = '12003'
format_image= '.jpg';
size_filter = 1;
std_filter  = 1;
std_noise   = 0.05;
for real=[1:1]
%Generate degraded image
f = imresize(double(imread(['original_images/',path,format_image]))/255.,1);
[m,n, ~] = size(f);
e_exacte = ones([m,n,2]).*(abs(D(f-1<0))>0);

% Filter 1
 fs     = (size_filter-1)/2;
[x,y]  = meshgrid(-fs:fs,-fs:fs);
arg    = -(x.*x + y.*y)/(2*std_filter*std_filter);
K = exp(arg);
%F(F<eps*max(F(:))) = 0;
K = K./sum(K(:));
fftK=psf2otf(K,[m,n]);
A= convop(fftK);
fBlurry = A * f;
noise = std_noise * randn(size(fBlurry)) ;
fNoisy = fBlurry + noise;
% fNoisy = imresize(double(imread(['degraded_images/small-mires-blur/noised_',name_image,'_noise_',num2str(std_noise),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.png']))/255.,1);
A_python = double(A);
name = ['degraded_images/',name_image,'_noise_',num2str(std_noise),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
save(name,'fNoisy','A','f','e_exacte','A_python');
end
