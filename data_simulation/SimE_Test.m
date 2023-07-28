%% Set the number of pairs you want to generate
num_pairs = 100;

%% Initialization
fp = '.\';
fn_aif = 'AIF_FDG.mat';
fn_xcat = 'XCAT_mask_tum.mat';
fn_lab = 'XCAT_mask_look_up_table.txt';

load([fp fn_aif],'aif')
load([fp fn_xcat],'xcat')
reg = get_XCAT_lab(fp,fn_lab);
for i=1:numel(reg)
    if isempty(reg(i).nam)
        reg(i)=reg(1); 
    end
end
iir = unique(xcat.dat(:));
xcat.atn = zeros(xcat.dim);
for i=2:numel(iir)
    xcat.atn(xcat.dat==iir(i))=reg(iir(i)).mu; 
end

%% Dynamic data
scan.n_bed = 6;
scan.t_bed = 30;                                                           % [s]
scan.t_del = 5;
scan.t_wb = scan.n_bed * ( scan.t_bed + scan.t_del );  
scan.t_start = 600;  
scan.n_fra = 16; nt = scan.n_fra; 
scan.n_phi = 240;
scan.fwhm = 6;                                                             % [mm]
scan.eff = 3e-3;

tt = ( scan.t_start + [ (0:nt-1)'*scan.t_wb, (0:nt-1)'*scan.t_wb + scan.t_bed ] ) / 60;

hdf5_filename = 'dataset_e_test.h5';
if isfile(hdf5_filename)
    delete(hdf5_filename)
end
tic
slices = 190:1:289; % Create a vector for the slice indices you want
motion_parameters = zeros(num_pairs, 3);  % initialize the matrix
for i = 1:num_pairs
    motion_parameters(i, :) = [2 + mod(i,14), -10+2*mod(i,11), 10-2*mod(i,11)];
end

for ipair=1:num_pairs
    %% Generate specific slice and motion parameters
    iz = slices(ipair); % Select the slice based on the current pair number
    mot.jt = motion_parameters(ipair, 1);  % Select the start time for motion based on your predefined array
    mot.xy = motion_parameters(ipair, 2:3);  % Select x and y motion displacements from your predefined array
    amp = 2;

    %% Create motion and motion-free pairs
    msk.dat = xcat.dat(:,:,iz);  
    iir = unique( msk.dat( msk.dat > 0 ) );  
    nr = numel( iir );

    img.dim = [ xcat.dim(1:2), scan.n_fra ];  img.vox = xcat.vox;  
    img.dat = zeros( img.dim );  img.atn = zeros( img.dim );
    img.Ki = zeros( img.dim(1:2) );  img.Vd = zeros( img.dim(1:2) );  
    img.Vb = zeros( img.dim(1:2) );

    for jt=1:scan.n_fra
        img.atn(:,:,jt) = xcat.atn(:,:,iz); 
    end

    for ir=1:nr
        par.kk = reg(iir(ir)).par(1:4)';  par.Vb = reg(iir(ir)).par(5);
        tac = sim_tac_2tc( aif, tt, par );
        aa = ( msk.dat == iir(ir) );
        for jt=1:scan.n_fra
            img.dat(:,:,jt) = img.dat(:,:,jt) + aa * tac.dat(jt);
        end
        ii = find( msk.dat == iir(ir) );
        img.Ki(ii) = par.kk(1) * par.kk(3) / sum(par.kk(2:3));
        img.Vd(ii) = par.kk(1) * par.kk(2) / sum(par.kk(2:3))^2;
        img.Vb(ii) = par.Vb;
    end

    %% Add noise & resolution
    sig = [ scan.fwhm / 2.355 ./ img.vox(1:2), 0 ];
    rec.dim = img.dim;
    rec.vox = img.vox;
    rec.dat = img.dat +  amp *  sqrt( img.dat ) .* randn( img.dim );
    rec.dat = conv_f( rec.dat, sig );
    rec.dat( rec.dat < 0 ) = 0;

    %% kinetic modelling
    PL_opt.g_ini = 0;  PL_opt.n_iter = 100;  
    rec.Ki_e = zeros( rec.dim(1:2) );
    rec.Vd_e = zeros( rec.dim(1:2) );

    tac.tt = tt;
    for ix=1:rec.dim(1)
        tac.dat = squeeze( rec.dat(ix,:,:) )';
        PL_e = fit_tac_PLe( aif, tac, PL_opt );
        rec.Ki_e(ix,:) = PL_e.Ki;
        rec.Vd_e(ix,:) = PL_e.Vd;
    end

    %% Save motion-free data
    motion_free_group = sprintf('/MotionFreeData_%d', ipair);
    
    params = {'Ki_e', 'Vd_e'};  % list of parameters
    
    for i = 1:length(params)
        param = params{i};
        dataset_path = [motion_free_group '/' param];
        data = rec.(param);
        h5create(hdf5_filename, dataset_path, size(data));
        h5write(hdf5_filename, dataset_path, data);
    end
    
    % Write attributes to the motion-free group.
    h5writeatt(hdf5_filename, motion_free_group, 'mot.jt', 0);
    h5writeatt(hdf5_filename, motion_free_group, 'mot.xy', [0, 0]);
    h5writeatt(hdf5_filename, motion_free_group, 'iz', iz);

    
    %% Add motion
    for jt=mot.jt:scan.n_fra
        rec.dat(:,:,jt) = circshift( rec.dat(:,:,jt), mot.xy); 
    end

    %% kinetic modelling for motion data
    for ix=1:rec.dim(1)
        tac.dat = squeeze( rec.dat(ix,:,:) )';
        PL_e = fit_tac_PLe( aif, tac, PL_opt );
        rec.Ki_e(ix,:) = PL_e.Ki;
        rec.Vd_e(ix,:) = PL_e.Vd;
    end

    %% Save motion data
    motion_group = sprintf('/MotionData_%d', ipair);

    for i = 1:length(params)
        param = params{i};
        dataset_path = [motion_group '/' param];
        data = rec.(param);
        h5create(hdf5_filename, dataset_path, size(data));
        h5write(hdf5_filename, dataset_path, data);
    end
    
    % Write attributes to the motion group.
    h5writeatt(hdf5_filename, motion_group, 'mot.jt', mot.jt);
    h5writeatt(hdf5_filename, motion_group, 'mot.xy', mot.xy);
    h5writeatt(hdf5_filename, motion_group, 'iz', iz);
end
toc
disp('<o>')

