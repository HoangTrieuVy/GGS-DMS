function dms_hohm(namefile)
load(['../degraded_images/',namefile,'.mat']);

[m,n] = size(f);
opts.verbose = 1;
opts.method = 'L2';
opts.isotropic = 0;
opts.groundTruth = f;
opts.muSeq = @(k) 2^k * 1e-6;


alpha_axis = linspace(3,-1,5);
gamma_axis = linspace(-5,1,5);

tab_Jaccard = zeros(length(alpha_axis),length(gamma_axis));
r = 0;
ind_alpha=1;
ind_gamma=1;
while(r<=3)
    dalpha = alpha_axis(1)-alpha_axis(2);
    dgamma = gamma_axis(2)-gamma_axis(1);
    alpha_axis_curr = 10.^alpha_axis;
    gamma_axis_curr = (10.^gamma_axis);
    for i= 1:length(alpha_axis_curr)
        for j=1:length(gamma_axis_curr)
            proxHandle = makeProxL2Linop( fNoisy, A);
            u = mumfordShah2D(gamma_axis_curr(j), alpha_axis_curr(i), proxHandle, opts);
            e = ones([m,n,2]).*(D(u).^2 > (gamma_axis_curr(j)/alpha_axis_curr(i)));
            tab_Jaccard(i,j)= jaccard(e,e_exacte);

        end
    end
    [M,I] = max(tab_Jaccard(:));

    [ind_alpha, ind_gamma] = ind2sub(size(tab_Jaccard),I);


    alpha_axis = linspace(alpha_axis(ind_alpha)+dalpha,alpha_axis(ind_alpha)-dalpha,5);
    gamma_axis = linspace(gamma_axis(ind_gamma)-dgamma,gamma_axis(ind_gamma)+dgamma,5);
    r= r+1;
end
maximum = max(max(tab_Jaccard));
out(n,m)= maximum;

proxHandle = makeProxL2Linop( fNoisy, A);
u_rec = mumfordShah2D(gamma_axis_curr(ind_gamma), alpha_axis_curr(ind_alpha), proxHandle, opts);
e_rec = ones([m,n,2]).*(D(u_rec).^2 > (gamma_axis_curr(ind_gamma)/alpha_axis_curr(ind_alpha)));
fprintf('Hohm:\t\t\t\t SNR = %3.2f\t SSIM=%3.2f\t jaccard=%3.3f\n',plpsnr(f*255,u_rec*255),ssim(f*255,u_rec*255),jaccard(e_rec,e_exacte));
save(['../results/Hohm_',namefile,'.mat'],'u_rec','e_rec')
