function [tac] = sim_tac_2tc( aif, tt, par )
%Simulate time-activity curves, 2TC

    nc = 2;
    nr = size(par.kk,2);  nt = size(tt,1);
    
    cb = zeros(nt,1);  cc = zeros(nt,nc*nr);

    %simulate blood curve
    for jt=1:nt
        iit = ( (aif.tt>tt(jt,1)) & (aif.tt<=tt(jt,2)) );
        cb(jt) = mean( aif.dat(iit) ); 
    end

    %Simulate time-activity curve
    cc0 = zeros(nc*nr,1);  opt = [];
    dt = (1/12);  tt1 = (0:dt:max(tt(:))+dt)';

    [~,cc1] = ode45( @ode_2tc4, tt1, cc0, opt );                           % generate curve

    for jt=1:nt
        iit = ( (tt1>tt(jt,1)) & (tt1<=tt(jt,2)) );
        cc(jt,:) = mean( cc1(iit,:) ); 
    end

    cc = reshape( cc, [nt,nc,nr] );
    
% Output

    tac.tt = tt;
    tac.dat = (1-par.Vb) * squeeze(sum( cc, 2 )) +... 
                 par.Vb  * cb * ones(1,nr);

% Nested function
    
    function [d_cc] = ode_2tc4( t1, cc )
    %2TC differential model
        cc = reshape( cc, [nc,nr] );
        Cp = interp1( aif.tt, aif.dat, t1 );
        d_c2 = par.kk(3,:).*cc(1,:) - par.kk(4,:).*cc(2,:);
        d_c1 = par.kk(1,:)*Cp       - par.kk(2,:).*cc(1,:) - d_c2;
        d_cc = reshape( [d_c1;d_c2], [nc*nr,1] );
    end
    
end
