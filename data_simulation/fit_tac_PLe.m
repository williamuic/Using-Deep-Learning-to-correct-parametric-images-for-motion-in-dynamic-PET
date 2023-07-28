function [PL,cc] = fit_tac_PLe( aif, tac, opt, PL )
%Fit TAC with Patlak model, MLEM

    nt = size( tac.dat, 1 );
    nc = size( tac.dat, 2 );
    s_aif = cumint( aif );
    ss = interp1( s_aif.tt, s_aif.dat, mean(tac.tt,2) );  n_ss = sum( ss );
    bb = interp1( aif.tt, aif.dat, mean(tac.tt,2) );      n_bb = sum( bb );
    
    if ( opt.g_ini )
        PL = fit_tac_PLg( aif, tac );
    elseif ( nargin < 4 )
        PL.Ki = 0.01 * ones(1,nc);  PL.Vd = 0.5 * ones(1,nc);
    end
    
    pp = zeros(2,nc);
    for ic=1:nc
        if ( max(tac.dat(:,ic)) > 0 )
            pp0 = [ PL.Ki(ic);  PL.Vd(ic) ];  pp0( pp0 < 0 ) = 0;
            pp(:,ic) = PL_mlem( pp0, tac.dat(:,ic) );
        end
    end
    
    PL.Ki = pp(1,:);  PL.Vd = pp(2,:);

    if ( nargout > 1 )
        cc = ( ss * PL.Ki + bb * PL.Vd )';
    end
   
    function [pp] = PL_mlem( pp0, dat )
        pp = pp0;
        for iter=1:opt.n_iter
            dd = pp(1) * ss + pp(2) * bb;
            d_cf = zeros(nt,1);
            ii = find( dd > 0 );
            d_cf(ii) = dat(ii) ./ dd(ii);
            cf = [ sum( d_cf .* ss ) / n_ss; sum( d_cf .* bb ) / n_bb ];
            pp = cf .* pp;
        end
    end
    
end

function [s_tac] = cumint( tac )
%Cumulated integral of TAC

    s_tac = tac;
    dt = tac.tt(2:end) - tac.tt(1:end-1);
    aa = ( tac.dat(1:end-1) + tac.dat(2:end) ) / 2;
    s_tac.dat(1) = 0;
    s_tac.dat(2:end) = cumsum( aa .* dt );

end
