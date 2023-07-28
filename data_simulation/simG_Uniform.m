%% Set the number of pairs you want to generate
iz_values = 150:349;  % Range of iz values
pairs_per_iz = 5;  % Number of pairs for each iz
num_pairs = length(iz_values) * pairs_per_iz;  % Total number of pairs

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

hdf5_filename = 'dataset_g_uniform.h5';
if isfile(hdf5_filename)
    delete(hdf5_filename)
end

tic
ipair = 1;
for iz = iz_values
    for pair_id = 1:pairs_per_iz
        %% Generate random noise level and motion parameters
        amp = 2;
        mot.jt = randi([2, scan.n_fra-1]);  % Randomly select a time frame for motion start
        mot.xy = randi([-10, 10], 1, 2);  % Randomly select x and y motion displacements between -10 and 10

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
        PL_opt.g_ini = 0;  PL_opt.n_iter = 250;  
        rec.Ki_g = zeros( rec.dim(1:2) );
        rec.Vd_g = zeros( rec.dim(1:2) );

        tac.tt = tt;
        for ix=1:rec.dim(1)
            tac.dat = squeeze( rec.dat(ix,:,:) )';
            PL_g = fit_tac_PLg( aif, tac );
            rec.Ki_g(ix,:) = PL_g.Ki;
            rec.Vd_g(ix,:) = PL_g.Vd;
        end

        %% Save motion-free data
        motion_free_group = sprintf('/MotionFreeData_%d', ipair);
        
        params = {'Ki_g', 'Vd_g'};  % list of parameters
        
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
            PL_g = fit_tac_PLg( aif, tac );
            rec.Ki_g(ix,:) = PL_g.Ki;
            rec.Vd_g(ix,:) = PL_g.Vd;
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
        
        %% Update pair count
        ipair = ipair + 1;
    end
end
toc
