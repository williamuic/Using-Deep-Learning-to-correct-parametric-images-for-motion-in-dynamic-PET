%% Initialization
fp = './';
fn_aif = 'AIF_FDG.mat';
fn_xcat = 'XCAT_mask_tum.mat';
%% Add 4D data for 3D PET images kinetic modelling first dimension 16 means 16 frames
images = 'images.mat';
motion_images = 'images_motion.mat';
%% Read in the parameters for kinetic modelling
fn_lab = 'XCAT_mask_look_up_table.txt';
load([fp sirf_images],'sirf_images')
load([fp fn_aif],'aif')
load([fp fn_xcat],'xcat')
reg = get_XCAT_lab(fp,fn_lab);
for i=1:numel(reg)
    if isempty(reg(i).nam)
        reg(i)=reg(1); 
    end
end
iir = unique(xcat.dat(:));
nr = numel( iir );
iir = unique( msk.dat( msk.dat > 0 ) );  
nr = numel( iir );
%% TODO: Need to convert img.dim as our new 3D simulation dim
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

%% Add noise & resolution (optional)
sig = [ scan.fwhm / 2.355 ./ img.vox(1:2), 0 ];
rec.dim = img.dim;
rec.vox = img.vox;
rec.dat = img.dat +  amp *  sqrt( img.dat ) .* randn( img.dim );
rec.dat = conv_f( rec.dat, sig );
rec.dat( rec.dat < 0 ) = 0;

%% kinetic modelling TODO:use 3D
PL_opt.g_ini = 0;  PL_opt.n_iter = 100;  
rec.Ki_e = zeros( rec.dim(1:2) );
rec.Vd_e = zeros( rec.dim(1:2) );
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
tac.tt = tt;
%% kinetic modelling
for ix=1:rec.dim(1)
    tac.dat = squeeze( rec.dat(ix,:,:) )';
    PL_e = fit_tac_PLe( aif, tac, PL_opt );
    rec.Ki_e(ix,:) = PL_e.Ki;
    rec.Vd_e(ix,:) = PL_e.Vd;
end

%% TODO:Save to hdf5 like shown in other scripts

