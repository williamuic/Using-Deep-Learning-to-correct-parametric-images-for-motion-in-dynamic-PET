%% Initialisation

fp = '.\';
fn_aif = 'AIF_FDG.mat';
fn_xcat = 'XCAT_mask_tum.mat';
fn_lab = 'XCAT_mask_look_up_table.txt';
load([fp fn_aif],'aif')
load([fp fn_xcat],'xcat')
reg = get_XCAT_lab(fp,fn_lab);
for i=1:numel(reg), if isempty(reg(i).nam), reg(i)=reg(1); end, end
iir = unique(xcat.dat(:));
xcat.atn = zeros(xcat.dim);
for i=2:numel(iir), xcat.atn(xcat.dat==iir(i))=reg(iir(i)).mu; end

figure(1), imagesc(squeeze(xcat.dat(:,200,:))'),colorbar
figure(2), imagesc(squeeze(xcat.atn(:,200,:))'),colorbar, colormap gray
figure(3), plot(aif.tt,aif.dat)

%% Dynamic data

scan.n_bed = 6;
scan.t_bed = 30;                                                           % [s]
scan.t_del = 5;
scan.t_wb = scan.n_bed * ( scan.t_bed + scan.t_del );  
scan.t_start = 600;  
scan.n_fra = 16;  nt = scan.n_fra;
scan.n_phi = 240;
scan.fwhm = 6;                                                             % [mm]
scan.eff = 3e-3;

tt = ( scan.t_start + [ (0:nt-1)'*scan.t_wb, (0:nt-1)'*scan.t_wb + scan.t_bed ] ) / 60;

iz = 208;  % 208
msk.dat = xcat.dat(:,:,iz);  
iir = unique( msk.dat( msk.dat > 0 ) );  nr = numel( iir );

img.dim = [ xcat.dim(1:2), nt ];  img.vox = xcat.vox;  
img.dat = zeros( img.dim );  img.atn = zeros( img.dim );
img.Ki = zeros( img.dim(1:2) );  img.Vd = zeros( img.dim(1:2) );  
img.Vb = zeros( img.dim(1:2) );

for jt=1:nt, img.atn(:,:,jt) = xcat.atn(:,:,iz); end

for ir=1:nr
    par.kk = reg(iir(ir)).par(1:4)';  par.Vb = reg(iir(ir)).par(5);
    tac = sim_tac_2tc( aif, tt, par );
    aa = ( msk.dat == iir(ir) );
    for jt=1:nt
        img.dat(:,:,jt) = img.dat(:,:,jt) + aa * tac.dat(jt);
    end
    ii = find( msk.dat == iir(ir) );
    img.Ki(ii) = par.kk(1) * par.kk(3) / sum(par.kk(2:3));
    img.Vd(ii) = par.kk(1) * par.kk(2) / sum(par.kk(2:3))^2;
    img.Vb(ii) = par.Vb;
end

figure(1), clf, colormap hot, imagesc(img.Ki'),colorbar
figure(2), clf, colormap hot, imagesc(img.Vd'),colorbar
figure(3), clf, colormap hot, imagesc(img.Vb'),colorbar
pause(1)
figure(1), clf, colormap hot
for jt=1:nt, imagesc(img.dat(:,:,jt)'),colorbar, pause(0.3), end

%% add noise & resolution
amp = 10;
sig = [ scan.fwhm / 2.355 ./ img.vox(1:2), 0 ];

rec.dim = img.dim;
rec.vox = img.vox;
rec.dat = img.dat +  amp *  sqrt( img.dat ) .* randn( img.dim );
rec.dat = conv_f( rec.dat, sig );
rec.dat( rec.dat < 0 ) = 0;

figure(1), clf, colormap hot
for jt=1:nt, imagesc(rec.dat(:,:,jt)'),colorbar, pause(0.2), end

%% add motion
mot.nn = 1;  mot.jt = rec.dim(3)/2;  mot.xy = [-5,0];

for jt=mot.jt:nt
    rec.dat(:,:,jt) = circshift( rec.dat(:,:,jt), mot.xy ); 
end

figure(1), clf, colormap hot
for jt=1:nt, imagesc(rec.dat(:,:,jt)'),colorbar, pause(0.2), end

%% kinetic modelling

PL_opt.g_ini = 0;  PL_opt.n_iter = 250;  
rec.Ki_g = zeros( rec.dim(1:2) );
rec.Vd_g = zeros( rec.dim(1:2) );
rec.Ki_e = zeros( rec.dim(1:2) );
rec.Vd_e = zeros( rec.dim(1:2) );

tac.tt = tt;
tic
for ix=1:rec.dim(1)
    tac.dat = squeeze( rec.dat(ix,:,:) )';
    PL_g = fit_tac_PLg( aif, tac );
    rec.Ki_g(ix,:) = PL_g.Ki;
    rec.Vd_g(ix,:) = PL_g.Vd;
    PL_e = fit_tac_PLe( aif, tac, PL_opt );
    rec.Ki_e(ix,:) = PL_e.Ki;
    rec.Vd_e(ix,:) = PL_e.Vd;
end
toc

figure(1), clf, colormap hot
figure(2), clf, colormap hot
figure(2), imagesc(rec.Vd_g'),colorbar, pause(1)
figure(1), imagesc(rec.Ki_e'),colorbar
figure(2), imagesc(rec.Vd_e'),colorbar, pause(1)
    
save(['Rec_' num2str(iz) '.mat'], 'rec')

disp('<o>')

