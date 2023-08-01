close all;
clear all;

% Step 1: Specify the folder containing the MAT files
folder_path = '../../../../Documents/dataset/BSDS500/data/param_dist_Jaccard/train';

% Step 2: Get a list of all files in the specified folder
file_list = dir(fullfile(folder_path, '*.mat'));
beta_list= [];
lambda_list= [];
% Step 3: Loop through the list and read each MAT file
for i = 1:length(file_list)
% for i = 1:1
    file_name = file_list(i).name;
    full_file_path = fullfile(folder_path, file_name);
    
    % Check if it's a file (not a directory) and has the .mat extension
    if ~file_list(i).isdir && endsWith(file_name, '.mat')
        try
            load(full_file_path);
            bmax=3;bmin=-1;lmax=2;lmin=-3;
            gridsize=5;
            [round,cs] = size(tab_coord_max_Jaccard_out);
            r=1;
            blist= linspace(bmax, bmin, gridsize);
            llist= linspace(lmin, lmax, gridsize);
            while r<=5   
                db = blist(1)-blist(2);
                dl = llist(2)-llist(1);
                coef_b_op_curr = blist(tab_coord_max_Jaccard_out(r,1)+1);
                coef_l_op_curr = llist(tab_coord_max_Jaccard_out(r,2)+1);
                b_op_curr = 10^blist(tab_coord_max_Jaccard_out(r,1)+1);
                l_op_curr = 10^llist(tab_coord_max_Jaccard_out(r,2)+1);
%                 disp(b_op_curr);
%                 disp(l_op_curr);
                blist= linspace(coef_b_op_curr+db, coef_b_op_curr-db, gridsize);
                llist= linspace(coef_l_op_curr-dl, coef_l_op_curr+dl, gridsize);
                r=r+1;
            end
            beta_list(end+1) = b_op_curr;
            lambda_list(end+1) = l_op_curr;
            
            fprintf('Loaded file: %s\n', file_name);
        catch
            fprintf('Error loading file: %s\n', file_name);
        end
    end
end


figure(1)
scatter(log10(lambda_list),log10(beta_list),'filled');
xlim([log10(min(lambda_list))-1 log10(max(lambda_list))+1]);
ylim([log10(min(beta_list))-1 log10(max(beta_list))+1]);
saveas(gcf, 'BSDS500_train_dist_param_Jaccard.png')



