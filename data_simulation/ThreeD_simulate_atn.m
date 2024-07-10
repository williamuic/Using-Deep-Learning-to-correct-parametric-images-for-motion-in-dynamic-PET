% Initialization of file paths and names
fp = '';  % Assuming file path is current directory, otherwise specify
fn_aif = 'AIF_FDG.mat';
fn_xcat = 'XCAT_mask_tum.mat';
fn_lab = 'XCAT_mask_look_up_table.txt';

% Load the arterial input function and xCAT data
load([fp fn_aif], 'aif');
load([fp fn_xcat], 'xcat');

% Get region labels from a lookup table
reg = get_XCAT_lab(fp, fn_lab);
for i = 1:numel(reg)
    if isempty(reg(i).nam)
        reg(i) = reg(1);
    end
end

% Process the XCAT data to generate attenuation coefficients
iir = unique(xcat.dat(:));
xcat.atn = zeros(xcat.dim);
for i = 2:numel(iir)
    xcat.atn(xcat.dat == iir(i)) = reg(iir(i)).mu;
end

% Visualize the loaded and processed data
figure(1), imagesc(squeeze(xcat.dat(:,200,:))'), colorbar
figure(2), imagesc(squeeze(xcat.atn(:,200,:))'), colorbar, colormap gray
figure(3), plot(aif.tt, aif.dat)

%% Dynamic data setup
scan.n_bed = 6;
scan.t_bed = 30;  % [s]
scan.t_del = 5;
scan.t_wb = scan.n_bed * (scan.t_bed + scan.t_del);  
scan.t_start = 600;  
scan.n_fra = 16;  nt = scan.n_fra;
scan.n_phi = 240;
scan.fwhm = 6;  % [mm]
scan.eff = 3e-3;

% Time intervals for the dynamic scan
tt = (scan.t_start + [(0:nt-1)' * scan.t_wb, (0:nt-1)' * scan.t_wb + scan.t_bed]) / 60;

% Prepare for 4D data simulation
num_slices = size(xcat.atn, 3);  % Total number of slices
img.dim = [256,256, nt, 150];  
img.atn = zeros(img.dim);

% Iterate over all slices and time points to fill img.atn and img.dat
for iz = 140:290  % Loop over slices
    for jt = 1:nt  % Loop over time points
        img.atn(:,:,jt,iz) = xcat.atn(73:328,73:328,iz);  % Populate attenuation data
        
        % Simulate img.dat similarly to img.atn for illustration
        % This assumes the simulation process for img.dat mirrors that of img.atn
        % Here, simply copying the xcat.dat data as a placeholder for actual simulation logic
    end
end
atn = img.atn
% Save the 4D arrays to MAT-files
save('img_atn_4D.mat', 'atn', '-v7.3');  % Saving with -v7.3 for potentially large data
