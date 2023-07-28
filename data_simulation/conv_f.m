function [bb] = conv_f( aa, sig )
%Fourier domain convolution with Gaussian PSF

    dim = size( aa );  dim1 = 1 + dim / 2;
    if ( numel(dim) == 2 )
        dim = [ dim, 1 ];
        dim1 = [ dim1, 1 ];
        sig = [ sig, 0 ];
    end
    
    [yy,xx,zz] = meshgrid( 1:dim(2), 1:dim(1), 1:dim(3) );
    uu = fftshift( xx - dim1(1) ) / dim(1);
    vv = fftshift( yy - dim1(2) ) / dim(2);
    ww = fftshift( zz - dim1(3) ) / dim(3);
    
    gg = exp( -2*pi^2 * ( sig(1)^2*uu.^2 + sig(2)^2*vv.^2 + sig(3)^2*ww.^2 ) );

    bb = real( ifftn( gg .* fftn( aa ) ) );
    
end
