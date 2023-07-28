function [PL,cc] = fit_tac_PLg( aif, tac )
%Fit TAC with Patlak graphical method

    tt1 = mean( tac.tt, 2 );
    s_aif = cumint( aif );
    ss = interp1( s_aif.tt, s_aif.dat, tt1 );
    bb = interp1( aif.tt, aif.dat, tt1 );
    xx = ss ./ bb;
    
    nc = size( tac.dat, 2 );
    kk = zeros(2,nc);
    for ic=1:nc
        if ( max(tac.dat(:,ic)) > 0 )
            yy = tac.dat(:,ic) ./ bb;
            kk(:,ic) = polyfit( xx, yy, 1 );
        end
    end
    
    PL.Ki = kk(1,:);  PL.Vd = kk(2,:);
    
    if ( nargout > 1 )
        cc = ( ss * PL.Ki + bb * PL.Vd )';
    end
    
%     figure(1)
%     x2 = max(xx);  y2 = max(yy);
%     plot(xx,yy,'ob',[0,x2],[0,x2]*PL.Ki+PL.Vd,'r-')
%     axis([0,x2,0,y2]*1.05)

end

function [s_tac] = cumint( tac )
%Cumulated integral of TAC

    s_tac = tac;
    dt = tac.tt(2:end) - tac.tt(1:end-1);
    aa = ( tac.dat(1:end-1) + tac.dat(2:end) ) / 2;
    s_tac.dat(1) = 0;
    s_tac.dat(2:end) = cumsum( aa .* dt );

end
