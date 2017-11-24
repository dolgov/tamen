% Expand a {d,R} TT decomposition into a full Matlab matrix.
%   function [Y_full] = expand_dR(Y)
%
% Computes the strong Kronecker products over the first cell dimension of X
% (expands the TT tensor into the full array), and sums over the second
% cell dimension.
%
%    !!! WARNING !!!! Curse of dimensionality !!! WARNING !!!!
% Use this routine only for small tensors.

function [Y] = expand_dR(y)
[d,n,Ry,ry,vectype] = grumble_vector(y,'y');
m = ones(d,1);
if (strcmp(vectype, 'tt_tensor'))
    y = cell2core(tt_matrix(y,n,1));
else
    % we might be OK about a matrix, but need to respect its sizes    
    for i=1:d
        if (issparse(y{i,1}))
            error('Expansion of sparse matrices is not allowed yet.');
        end
        [~,n(i),m(i),~] = size(y{i,1});
    end
end;


for k=1:Ry
    Yk = 1;
    ncum = 1;
    mcum = 1;
    for i=1:d
        Xi = reshape(y{i,k}, ry(i,k), n(i)*m(i)*ry(i+1,k));
        Yk = Yk*Xi;
        Yk = reshape(Yk, ncum, mcum, n(i), m(i)*ry(i+1,k));
        Yk = permute(Yk, [1,3,2,4]);
        ncum = ncum*n(i);
        mcum = mcum*m(i);
        Yk = reshape(Yk, ncum*mcum, ry(i+1,k));
    end
    Yk = reshape(Yk, ncum, mcum, []);
    if (k==1)
        Y = Yk;
    else
        Y = Y + Yk;
    end;
end
end
