clear all;
clc;
setPath;
%Parameters of degraded image
% name_image= 'dots_52_v3';
% name_image='24004';
name_image='35008';
% name_image='118035';
% name_image='12003';
% name_image='35010';
% name_image='65019';
% name_image='124084';

% format_image= '.png';
format_image= '.jpg';
size_filter = 1;
std_filter  = 1;
std_noise   = 0.05;
real=1;
nb_iter =300;
% Load degraded data
name = ['degraded_images/',name_image,'_noise_',num2str(std_noise),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
load(name);
figure(3);
colormap(gray);
subplot(2,4,1);imagesc(f);plot_contours(e_exacte); title 'Original';
subplot(2,4,2);imagesc(fNoisy);axis image off; title({'Degraded';[num2str(size_filter),'\',num2str(std_filter),'\',num2str(std_noise)];['PSNR: ' num2str(plpsnr(f,fNoisy))]} );
disp(name);
fprintf('Noisy:\t\t\t\t SNR = %3.2f\t SSIM=%3.2f\t \n',plpsnr(f*255,fNoisy*255),ssim(f*255,fNoisy*255));

% Load results TROF
name = ['results/TROF_',int2str(nb_iter),'_',name_image,'_noise_',num2str(std_noise),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
load(name);
fprintf('TROF:\t\t\t\t SNR = %3.2f\t SSIM=%3.2f\t jaccard=%3.3f\n',plpsnr(f*255,u_rec*255),ssim(f*255,u_rec*255),jaccard(e_rec,e_exacte));
subplot(2,4,3);imagesc(u_rec,[0,1]);plot_contours(e_rec); title({'T-ROF ';['PSNR: ' num2str(plpsnr(f,u_rec),'%4.2f')];[ ' Jaccard: ' num2str(jaccard(e_rec,e_exacte),'%3.3f')]});

% Load results Hohm
name = ['results/Hohm_',name_image,'_noise_',num2str(std_noise),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
load(name);
fprintf('Hohm:\t\t\t\t SNR = %3.2f\t SSIM=%3.2f\t jaccard=%3.3f\n',plpsnr(f*255,u_rec*255),ssim(f*255,u_rec*255),jaccard(e_rec,e_exacte));
subplot(2,4,4);imagesc(u_rec,[0,1]);plot_contours(e_rec); title({'Hohm et al';['PSNR: ' num2str(plpsnr(f,u_rec),'%4.2f')];[ ' Jaccard: ' num2str(jaccard(e_rec,e_exacte),'%3.3f')]});

% % Load results SLPAM-l1
% name = ['results/SL-PAM_l1_',int2str(nb_iter),'_',name_image,'_noise_',num2str(std_noise),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
% load(name);
% fprintf('SLPAM_l1:\t\t\t SNR = %3.2f\t SSIM=%3.2f\t jaccard=%3.3f\n',plpsnr(f*255,u_rec*255),ssim(f*255,u_rec*255),jaccard(e_rec,e_exacte));
% subplot(2,4,5);imagesc(u_rec);plot_contours(e_rec); title({'SLPAM l1 ';['PSNR: ' num2str(plpsnr(f,u_rec),'%4.2f')];[ ' Jaccard: ' num2str(jaccard(e_rec,e_exacte),'%3.3f')]});

% Load results PALM-l1
name = ['results/PALM_l1_',int2str(nb_iter),'_',name_image,'_noise_',num2str(std_noise),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
load(name);
fprintf('PALM_l1:\t\t\t SNR = %3.2f\t SSIM=%3.2f\t jaccard=%3.3f\n',plpsnr(f*255,u_rec*255),ssim(f*255,u_rec*255),jaccard(e_rec,e_exacte));
subplot(2,4,6);imagesc(u_rec,[0,1]);plot_contours(e_rec); title({'PALM l1';['PSNR: ' num2str(plpsnr(f,u_rec),'%4.2f')];[ ' Jaccard: ' num2str(jaccard(e_rec,e_exacte),'%3.3f')]});

% % Load results PALM-AT-fixed
% name = ['results/PALM_AT-fourier_',int2str(nb_iter),'_',name_image,'_noise_',num2str(std_noise),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
% load(name);
% fprintf('PALM_AT fixed:\t\t\t SNR = %3.2f\t SSIM=%3.2f\t jaccard=%3.3f\n',plpsnr(f*255,u_rec*255),ssim(f*255,u_rec*255),jaccard(e_rec,e_exacte));
% subplot(2,4,7);imagesc(u_rec);plot_contours(e_rec); title({'PALM AT fixed';['PSNR: ' num2str(plpsnr(f,u_rec),'%4.2f')];[ ' Jaccard: ' num2str(jaccard(e_rec,e_exacte),'%3.3f')]});

% Load results PALM-AT-eps-descent
name = ['results/PALM-eps-descent_AT-fourier_',int2str(nb_iter),'_',name_image,'_noise_',num2str(std_noise),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
load(name);
fprintf('PALM-AT-descent:\t SNR = %3.2f\t SSIM=%3.2f\t jaccard=%3.3f\n',plpsnr(f*255,u_rec*255),ssim(f*255,u_rec*255),jaccard(e_rec,e_exacte));
subplot(2,4,8);imagesc(u_rec,[0,1]);plot_contours(e_rec); title({'PALM AT descent';['PSNR: ',num2str(plpsnr(f,u_rec),'%4.2f')];[' Jaccard: ' num2str(jaccard(e_rec,e_exacte),'%3.3f')]});
