clear all;
clc;
setPath;
%Parameters of degraded image
name_image= 'dots_52_v3';
format_image= '.png';
size_filter = 1;
std_filter  = 1;
std_noise   = [0.02,0.08,0.1,0.2];
nb_real=10;
for n   = 1:length(std_noise)
    for real=1:nb_real
    nb_iter =300;
    % Load degraded data
    name = ['degraded_images/',name_image,'_noise_',num2str(std_noise(n)),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
    load(name);% figure(1);
    perf_noisy(1,real,n)=plpsnr(f*255,fNoisy*255);
    perf_noisy(2,real,n)=ssim(f*255,fNoisy*255);

    % Load results TROF
    name = ['results/TROF_',int2str(nb_iter),'_',name_image,'_noise_',num2str(std_noise(n)),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
    load(name);
    perf_TROF(1,real,n)=plpsnr(f*255,u_rec*255);
    perf_TROF(2,real,n)=ssim(f*255,u_rec*255);
    perf_TROF(3,real,n)=jaccard(e_rec,e_exacte);


    % Load results Hohm
    name = ['results/Hohm_',name_image,'_noise_',num2str(std_noise(n)),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
    load(name);
    perf_hohm(1,real,n)=plpsnr(f*255,u_rec*255);
    perf_hohm(2,real,n)=ssim(f*255,u_rec*255);
    perf_hohm(3,real,n)=jaccard(e_rec,e_exacte);


    % Load results PALM-l1
    name = ['results/PALM_l1_',int2str(nb_iter),'_',name_image,'_noise_',num2str(std_noise(n)),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
    load(name);
    perf_PALM_l1(1,real,n)=plpsnr(f*255,u_rec*255);
    perf_PALM_l1(2,real,n)=ssim(f*255,u_rec*255);
    perf_PALM_l1(3,real,n)=jaccard(e_rec,e_exacte);
% 
%     name = ['results/SL-PAM_l1_',int2str(nb_iter),'_',name_image,'_noise_',num2str(std_noise(n)),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
%     load(name);
%     perf_SLPAM_l1(1,real,n)=plpsnr(f*255,u_rec*255);
%     perf_SLPAM_l1(2,real,n)=ssim(f*255,u_rec*255);
%     perf_SLPAM_l1(3,real,n)=jaccard(e_rec,e_exacte);

% 
%     name = ['results/PALM_AT-fourier_',int2str(nb_iter),'_',name_image,'_noise_',num2str(std_noise(n)),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
%     load(name);
%     perf_PALM_AT_fixed(1,real,n)=plpsnr(f*255,u_rec*255);
%     perf_PALM_AT_fixed(2,real,n)=ssim(f*255,u_rec*255);
%     perf_PALM_AT_fixed(3,real,n)=jaccard(e_rec,e_exacte);

    % Load results PALM-AT-eps-descent
    name = ['results/PALM-eps-descent_AT-fourier_',int2str(nb_iter),'_',name_image,'_noise_',num2str(std_noise(n)),'_blur_',int2str(size_filter),'_',num2str(std_filter),'_',int2str(real),'.mat'];
    load(name);
    perf_PALM_AT_descent(1,real,n)=plpsnr(f*255,u_rec*255);
    perf_PALM_AT_descent(2,real,n)=ssim(f*255,u_rec*255);
    perf_PALM_AT_descent(3,real,n)=jaccard(e_rec,e_exacte);

    fprintf('\n\n\n')
    end
end

%% Display 
figure(1);clf;
h=hsv(10);
subplot(1,3,1);
plot(std_noise,1/nb_real*squeeze(sum(perf_noisy(1,:,:),2)),'Color',h(1,:));hold on;
plot(std_noise,1/nb_real*squeeze(sum(perf_TROF(1,:,:),2)),'Color',h(2,:));hold on;
plot(std_noise,1/nb_real*squeeze(sum(perf_hohm(1,:,:),2)),'Color','k');hold on;
% plot(std_noise,1/nb_real*squeeze(sum(perf_SLPAM_l1(1,:,:),2)),'Color',h(4,:));hold on;
plot(std_noise,1/nb_real*squeeze(sum(perf_PALM_l1(1,:,:),2)),'Color',h(5,:));hold on;
% plot(std_noise,1/nb_real*squeeze(sum(perf_PALM_AT_fixed(1,:,:),2)),'Color',h(6,:));hold on;
plot(std_noise,1/nb_real*squeeze(sum(perf_PALM_AT_descent(1,:,:),2)),'Color',h(7,:));hold on;
grid on;
title('PNSR');
% legend('Noisy','T-ROF','Hohm et al', 'SL-PAM l1','PALM l1','PALM-AT fixed','PALM-AT descent')
% legend('Noisy','T-ROF','Hohm et al', 'SL-PAM l1','PALM l1','PALM-AT descent')
legend('Noisy','T-ROF','Hohm et al','PALM l1','PALM-AT descent')

subplot(1,3,2);
plot(std_noise,1/nb_real*squeeze(sum(perf_noisy(2,:,:),2)),'Color',h(1,:));hold on;
plot(std_noise,1/nb_real*squeeze(sum(perf_TROF(2,:,:),2)),'Color',h(2,:));hold on;
plot(std_noise,1/nb_real*squeeze(sum(perf_hohm(2,:,:),2)),'Color','k');hold on;
% plot(std_noise,1/nb_real*squeeze(sum(perf_SLPAM_l1(2,:,:),2)),'Color',h(4,:));hold on;
plot(std_noise,1/nb_real*squeeze(sum(perf_PALM_l1(2,:,:),2)),'Color',h(5,:));hold on;
% plot(std_noise,1/nb_real*squeeze(sum(perf_PALM_AT_fixed(2,:,:),2)),'Color',h(6,:));hold on;
plot(std_noise,1/nb_real*squeeze(sum(perf_PALM_AT_descent(2,:,:),2)),'Color',h(7,:));hold on;
grid on;
% legend('Noisy','T-ROF','Hohm et al', 'SL-PAM l1','PALM l1','PALM-AT fixed','PALM-AT descent')
title('SSIM');

subplot(1,3,3);
plot(std_noise,1/nb_real*squeeze(sum(perf_TROF(3,:,:),2)),'Color',h(2,:));hold on;
plot(std_noise,1/nb_real*squeeze(sum(perf_hohm(3,:,:),2)),'Color','k');hold on;
% plot(std_noise,1/nb_real*squeeze(sum(perf_SLPAM_l1(3,:,:),2)),'Color',h(4,:));hold on;
plot(std_noise,1/nb_real*squeeze(sum(perf_PALM_l1(3,:,:),2)),'Color',h(5,:));hold on;
% plot(std_noise,1/nb_real*squeeze(sum(perf_PALM_AT_fixed(3,:,:),2)),'Color',h(6,:));hold on;
plot(std_noise,1/nb_real*squeeze(sum(perf_PALM_AT_descent(3,:,:),2)),'Color',h(7,:));hold on;
grid on;
% legend('T-ROF','Hohm et al', 'SL-PAM l1','PALM l1','PALM-AT fixed','PALM-AT descent')
title('Jaccard');







