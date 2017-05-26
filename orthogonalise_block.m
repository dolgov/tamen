% Orthogonalises a TT block and (optionally) casts the R-factor onto a
% neighbouring block
%   function [x_next, x, rx] = orthogonalise_block(x_next, x, dir)
%
% !!! Note !!!!
% This is a technical routine, please use higher-level amen_solve or tamen 
% unless you understand precisely what are you doing.

function [x_next, x, rx] = orthogonalise_block(x_next, x, dir)
if (dir>0)
    % Forward
    [rx1,n1,~,rx] = size(x);
    x = reshape(x, rx1*n1, rx);
    [x, rv]=qr(x, 0);
    if (~isempty(x_next))
        [~,n2,~,rx2] = size(x_next);
        x_next = reshape(x_next, rx, n2*rx2);
        x_next = rv*x_next;
        x_next = reshape(x_next, [], n2, 1, rx2); % rx (new)
    end;
    rx = size(x, 2);
    x = reshape(x, rx1, n1, 1, rx);
else
    % Backward
    [rx,n1,~,rx2] = size(x);
    x = reshape(x, rx, n1*rx2);
    x = x.';
    [x, rv]=qr(x, 0);
    if (~isempty(x_next))
        [rx0,n0,~,~] = size(x_next);
        x_next = reshape(x_next, rx0*n0, rx);
        x_next = x_next*rv.';
        x_next = reshape(x_next, rx0, n0, 1, []); % rx(new)
    end;
    rx = size(x, 2);
    x = x.';
    x = reshape(x, rx, n1, 1, rx2);
end;
end
