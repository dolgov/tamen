% Chebyshev differentiation matrix and grid.
%   function [t,S]=chebdiff(N)
% 
% Computes the spectral Chebyshev differentiation matrix and nodes
% on the interval (0,1]. Used Dirichlet BC at 0, and t is sorted acsending.
% See [L. N. Trefethen, Spectral Methods in Matlab, SIAM, Philadelphia, 2000].

function [t,S]=chebdiff(N)
% Generate nodes
x = ((N-1):-1:0)';
t = 0.5*(cos(pi*x/N)+1);
if (nargout>1)
    % Generate the matrix
    c = [ones(N-1,1); 2].*(-1).^x;
    T = repmat(t,1,N);
    T = T-T';
    S = (c*(1./c)')./(T+eye(N));
    S = S-diag(sum(S, 2)+0.5*(-1)^N*(c./t));
end;
end
