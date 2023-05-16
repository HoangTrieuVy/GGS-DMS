clear all;
clc;
% addpath('cbrewer')
close all;

zx=64;zy=64;scale=2;
% 
% rect= [137,60,zx,zy];
% nameim="101085";

% rect= [200,100,zx,zy];
% nameim="101087";
% 
rect= [100,100,zx,zy];
nameim="102061";
% 
% rect= [200,100,zx,zy];
% nameim="103070";


load('SLPAML1_IR_102061.mat')

imwrite(res, fullfile("res_SLPAML1_"+nameim+".png"))
imwrite(x_ref, fullfile("original_102061.png"))
imwrite(y_ref, fullfile("corrupted_102061.png"))



zx=128;zy=128;scale=2;
rect= [100,100,zx,zy];
roi_xest = imcrop(res, rect);
zoom_roi=imresize(roi_xest,2);
nameim='102061';
imwrite(imresize(zoom_roi, 2), fullfile("zoom_res_SLPAML1"+nameim+".png"))        



