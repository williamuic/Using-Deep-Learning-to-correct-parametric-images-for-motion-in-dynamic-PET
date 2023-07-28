function [reg,t_typ] = get_XCAT_lab( fp, fn )
%Get XCAT regions and kinetic parameters

    reg=[];
    fid = fopen([fp fn],'r');
    if ( fid < 0 ), reg=[]; disp('Error opening file'), return, end
    
    for i=1:5, lin=fgetl(fid); end
    while(1)
        lin = fgetl( fid );
        if isempty(lin), break, end
        ss = strsplit( lin, '=' );    nam = ss{1};
        ss = strsplit( ss{2}, '#' );  ind = str2num( ss{1} );
        ss = strsplit( ss{2}, ';' );  par = str2num( ss{2} );
        reg(ind).nam = nam;
        reg(ind).com = ss{1};
        reg(ind).par = par;
    end

    t_typ=[];
    lin = fgetl(fid);
    for jt=1:100
        lin = fgetl( fid );
        if isempty(lin), break, end
        ss = strsplit( lin, '=' );  
        t_typ(jt).nam = strtrim( ss{1} );
        t_typ(jt).mu = str2num( ss{2} ) / 10;
    end
    fclose( fid );

    for ir=1:numel(reg)
        if ~isempty( reg(ir).nam )
            reg(ir).mu = t_typ(1).mu;
            for jt=1:numel(t_typ)
                t_nam = lower( strsplit( t_typ(jt).nam, ' ' ) );
                if ( contains( reg(ir).nam, t_nam{1} ) ||...
                     contains( reg(ir).nam, t_nam{end} ) ||...
                     contains( reg(ir).com, t_nam{1} ) ||...
                     contains( reg(ir).com, t_nam{end} ) )
                    reg(ir).mu = t_typ(jt).mu;
                end
            end
        end
    end

end
