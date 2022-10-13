clear all;
clc;
setPath;
%Parameters of degraded image
name_image= 'dots_52_v3';
format_image= '.png';
size_filter = 1;
std_filter  = 1;
std_noise   = 0.05;
for real=1:10
nb_iter =300;
% Load degraded data
name = ['degraded_images/',name_image,'_noise_',num2str(std_noise),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
load(name);% figure(1);
% colormap(gray);
% subplot(2,3,1);imagesc(f);plot_contours(e_exacte); title 'Original';
% subplot(2,3,2);imagesc(fNoisy);axis image off; title({'Degraded';[num2str(size_filter),'\',num2str(std_filter),'\',num2str(std_noise)];['PSNR: ' num2str(plpsnr(f,fNoisy))]} );
disp(name);
perf_noisy(1,real)=plpsnr(f*255,fNoisy*255);
perf_noisy(2,real)=ssim(f*255,fNoisy*255);
fprintf('Noisy:\t\t\t\t SNR = %3.2f\t SSIM=%3.2f\t \n',median(perf_noisy(1,:)),median(perf_noisy(2,:)));

% Load results TROF
name = ['results/TROF_',int2str(nb_iter),'_',name_image,'_noise_',num2str(std_noise),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
load(name);
% subplot(2,3,3);imagesc(u_rec);plot_contours(e_rec); title({'T-ROF ';['PSNR: ' num2str(plpsnr(f,u_rec),'%4.2f')];[ ' Jaccard: ' num2str(jaccard(e_rec,e_exacte),'%3.3f')]});
perf_TROF(1,real)=plpsnr(f*255,u_rec*255);
perf_TROF(2,real)=ssim(f*255,u_rec*255);
perf_TROF(3,real)=jaccard(e_rec,e_exacte);
fprintf('TROF:\t\t\t\t SNR = %3.2f\t SSIM=%3.2f\t jaccard=%3.2f\n',median(perf_TROF(1,:)),median(perf_TROF(2,:)),median(perf_TROF(3,:)));


% Load results Hohm
name = ['results/Hohm_',name_image,'_noise_',num2str(std_noise),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
load(name);
% subplot(2,3,4);imagesc(u_rec);plot_contours(e_rec); title({'Hohm et al';['PSNR: ' num2str(plpsnr(f,u_rec),'%4.2f')];[ ' Jaccard: ' num2str(jaccard(e_rec,e_exacte),'%3.3f')]});
perf_hohm(1,real)=plpsnr(f*255,u_rec*255);
perf_hohm(2,real)=ssim(f*255,u_rec*255);
perf_hohm(3,real)=jaccard(e_rec,e_exacte);
fprintf('Hohm:\t\t\t\t SNR = %3.2f\t SSIM=%3.2f\t jaccard=%3.2f\n',median(perf_hohm(1,:)),median(perf_hohm(2,:)),median(perf_hohm(3,:)));


% Load results PALM-l1
name = ['results/PALM_l1_',int2str(nb_iter),'_',name_image,'_noise_',num2str(std_noise),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
load(name);
% subplot(2,3,5);imagesc(u_rec);plot_contours(e_rec); title({'PALM l1';['PSNR: ' num2str(plpsnr(f,u_rec),'%4.2f')];[ ' Jaccard: ' num2str(jaccard(e_rec,e_exacte),'%3.3f')]});
perf_PALM_l1(1,real)=plpsnr(f*255,u_rec*255);
perf_PALM_l1(2,real)=ssim(f*255,u_rec*255);
perf_PALM_l1(3,real)=jaccard(e_rec,e_exacte);
fprintf('PALM_l1:\t\t\t SNR = %3.2f\t SSIM=%3.2f\t jaccard=%3.2f\n',median(perf_PALM_l1(1,:)),median(perf_PALM_l1(2,:)),median(perf_hohm(3,:)));

name = ['results/SL-PAM_l1_',int2str(nb_iter),'_',name_image,'_noise_',num2str(std_noise),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
load(name);
perf_SLPAM_l1(1,real)=plpsnr(f*255,u_rec*255);
perf_SLPAM_l1(2,real)=ssim(f*255,u_rec*255);
perf_SLPAM_l1(3,real)=jaccard(e_rec,e_exacte);
fprintf('SL-PAM_l1:\t\t\t SNR = %3.2f\t SSIM=%3.2f\t jaccard=%3.2f\n',median(perf_SLPAM_l1(1,:)),median(perf_SLPAM_l1(2,:)),median(perf_SLPAM_l1(3,:)));


name = ['results/PALM_AT-fourier_',int2str(nb_iter),'_',name_image,'_noise_',num2str(std_noise),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
load(name);
perf_PALM_AT_fixed(1,real)=plpsnr(f*255,u_rec*255);
perf_PALM_AT_fixed(2,real)=ssim(f*255,u_rec*255);
perf_PALM_AT_fixed(3,real)=jaccard(e_rec,e_exacte);
fprintf('PALM-AT-fixed:\t\t\t SNR = %3.2f\t SSIM=%3.2f\t jaccard=%3.2f\n',median(perf_PALM_AT_fixed(1,:)),median(perf_PALM_AT_fixed(2,:)),median(perf_PALM_AT_fixed(3,:)));

% Load results PALM-AT-eps-descent
name = ['results/PALM-eps-descent_AT-fourier_',int2str(nb_iter),'_',name_image,'_noise_',num2str(std_noise),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
load(name);
% subplot(2,3,6);imagesc(u_rec);plot_contours(e_rec); title({'PALM AT descent';['PSNR: ',num2str(plpsnr(f,u_rec),'%4.2f')];[' Jaccard: ' num2str(jaccard(e_rec,e_exacte),'%3.3f')]});
perf_PALM_AT_descent(1,real)=plpsnr(f*255,u_rec*255);
perf_PALM_AT_descent(2,real)=ssim(f*255,u_rec*255);
perf_PALM_AT_descent(3,real)=jaccard(e_rec,e_exacte);
fprintf('PALM-AT-descent:\t\t SNR = %3.2f\t SSIM=%3.2f\t jaccard=%3.2f\n',median(perf_PALM_AT_descent(1,:)),median(perf_PALM_AT_descent(2,:)),median(perf_PALM_AT_descent(3,:)));

fprintf('\n\n\n')

end
