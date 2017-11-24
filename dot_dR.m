% Dot product of two {d,R} TT decompositions, S = X'*Y.
%   function [S] = dot_dR(X,Y)
%
% X and Y should be dense.

function [S] = dot_dR(x,y)
[d,n,Rx,rx,vectype] = grumble_vector(x,'y');
if (strcmp(vectype, 'tt_tensor'))
    x = cell2core(tt_matrix(x,n,1));
end
[dy,ny,Ry,ry,vectype] = grumble_vector(y,'y');
if (d~=dy)||(any(n~=ny))
    error('Sizes of x and y should be the same');
end
if (strcmp(vectype, 'tt_tensor'))
    y = cell2core(tt_matrix(y,n,1));
end

S = 0;
for m=1:Ry
    for k=1:Rx
        Skm = 1;
        for i=1:d
            Skm = Skm*reshape(y{i,m}, ry(i,m), n(i)*ry(i+1,m));
            Skm = reshape(Skm, rx(i,k)*n(i), ry(i+1,m));
            Skm = reshape(x{i,k}, rx(i,k)*n(i), rx(i+1,k))'*Skm;
        end
        S = S + Skm;
    end
end
end
