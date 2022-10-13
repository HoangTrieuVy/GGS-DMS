function [jacc,j] = jaccard(est,gt,sig)
% est = (estx,esty) : caracteristic function of the estimated contour
% gt = (gtx,gty) : ground truth
% sig : variance of the gaussian blur


%gt  = double(gt > 0.5);
%est = double(est > 0.5);

% figure(1001), clf;
% subplot(221), imshow(gt,[]);
% subplot(222), imshow(est,[]);
% subplot(223), imshow(cat(3,gt,est,gt-gt),[]);
        

inter = sum(double(gt(:) == 1. & est(:) == 1.));
union = sum(double(gt(:) == 0. & est(:) == 1.)) ...
      + sum(double(gt(:) == 1. & est(:) == 0.)) ...
      + sum(double(gt(:) == 1. & est(:) == 1.)) ;

if union == 0.
    jacc = 0.;
else
    jacc = inter./union;
end










%% not a good way !
% gt = double(gt > 0.5);
% gtx = gt(:,:,1);
% gty = gt(:,:,2);
% 
% est = double(est > 0.5);
% estx = est(:,:,1);
% esty = est(:,:,2);
% 
% inter_x = sum(double(gtx(:) == 1. & estx(:) == 1.));
% union_x = sum(double(gtx(:) == 0. & estx(:) == 1.)) ...
%         + sum(double(gtx(:) == 1. & estx(:) == 0.)) ...
%         + sum(double(gtx(:) == 1. & estx(:) == 1.)) ;
% 
% inter_y = sum(double(gty(:) == 1. & esty(:) == 1.));
% union_y = sum(double(gty(:) == 0. & esty(:) == 1.)) ...
%         + sum(double(gty(:) == 1. & esty(:) == 0.)) ...
%         + sum(double(gty(:) == 1. & esty(:) == 1.)) ;
% 
% 
% j = .5 .* (inter_x./union_x + inter_y./union_y);

end