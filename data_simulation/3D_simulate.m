%% Initialisation

fp = '';
fn_aif = 'AIF_FDG.mat';
fn_xcat = 'XCAT_mask_tum.mat';
fn_lab = 'XCAT_mask_look_up_table.txt';
load([fp fn_aif],'aif')
load([fp fn_xcat],'xcat')
reg = get_XCAT_lab(fp,fn_lab);
for i=1:numel(reg), if isempty(reg(i).nam), reg(i)=reg(1); end, end
iir = unique(xcat.dat(:));

% Intercept the middle 256x256x127 region from indices 164:290 along the z-axis
mid_x = 256;
mid_y = 256;
mid_z_start = 164;
mid_z_end = 290;

start_x = floor((xcat.dim(1) - mid_x) / 2) + 1;
start_y = floor((xcat.dim(2) - mid_y) / 2) + 1;

xcat.dat = xcat.dat(start_x:start_x + mid_x - 1, start_y:start_y + mid_y - 1, mid_z_start:mid_z_end);
xcat.dim = size(xcat.dat);

xcat.atn = zeros(xcat.dim);
for i=2:numel(iir), xcat.atn(xcat.dat==iir(i))=reg(iir(i)).mu; end

figure(1), imagesc(squeeze(xcat.dat(:,round(end/2),:))'),colorbar
figure(2), imagesc(squeeze(xcat.atn(:,round(end/2),:))'),colorbar, colormap gray
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

msk.dat = xcat.dat(:,:,:);  
iir = unique( msk.dat( msk.dat > 0 ) );  nr = numel( iir );

img.dim = [ xcat.dim, nt ];  img.vox = xcat.vox;  
img.dat = zeros( img.dim );  img.atn = zeros( img.dim );
img.Ki = zeros( xcat.dim );  img.Vd = zeros( xcat.dim );  
img.Vb = zeros( xcat.dim );

for jt=1:nt, img.atn(:,:,:,jt) = xcat.atn(:,:,:); end

for ir=1:nr
    par.kk = reg(iir(ir)).par(1:4)';  par.Vb = reg(iir(ir)).par(5);
    tac = sim_tac_2tc( aif, tt, par );
    aa = ( msk.dat == iir(ir) );
    for jt=1:nt
        img.dat(:,:,:,jt) = img.dat(:,:,:,jt) + aa * tac.dat(jt);
    end
    ii = find( msk.dat == iir(ir) );
    img.Ki(ii) = par.kk(1) * par.kk(3) / sum(par.kk(2:3));
    img.Vd(ii) = par.kk(1) * par.kk(2) / sum(par.kk(2:3))^2;
    %img.Vb(ii) = par.Vb;
end
%%
emi = img.dat;
atn = img.atn;
save('emi_data.mat', 'emi', '-v7.3');
save('atn_data.mat', 'atn', '-v7.3');

%%
figure(1), clf, colormap hot, imagesc(squeeze(img.Ki(:,:,60))'),colorbar
figure(2), clf, colormap hot, imagesc(squeeze(img.Vd(:,:,60))'),colorbar
%%
pause(1)
figure(1), clf, colormap hot
for jt=1:nt, imagesc(squeeze(img.dat(:,128,:,jt))'),colorbar, pause(0.3), end
