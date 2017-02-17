% Interpolation on a Chebyshev grid in time
%   function [x]=extract_snapshot(X,t,tx)
%
% Extracts x=X(:,...,:,tx) from tamen outputs X (solution TT tensor(s)) and 
% t (cell array of time grid points) via barycentric Chebyshev interpolation. 
% For stability, tx should be in [0,1].
% If X is in {d+1,R} format, returns x in {d,numel(tx)} format.
% If X is a tt_tensor (d+1-dimensional), returns x as a d-dimensional
% tt_tensor, with the last border rank equal to numel(tx).

function [x]=extract_snapshot(X,t,tx)
if (isa(X,'tt_tensor'))
    d = X.d-1;
    x = chunk(X,1,d);
    Xt = X{d+1};
else
    d = size(X,1)-1;
    Xt = X(d+1,:);
end;

if (numel(tx)==1)&&(tx==1)
    % Process this special case separately
    if (isa(X,'tt_tensor'))
        x = x*Xt(:,end);
    else
        x = X(1:d, end);
        [rd,nd,md,rdp]=size(x{d});
        x{d} = reshape(x{d}, rd*nd*md, rdp);
        x{d} = x{d}*Xt{end}(:,end);
        x{d} = reshape(x{d}, rd, nd, md);
    end;
    return;
end;

%%%% General t_needed (can be a vector also)
if (isa(t,'double'))
    t = {t};
end;
% number of grid points in each subinterval
nt = cellfun(@numel, t);
% find which interval t_needed belongs to
ind = ones(numel(tx), 1);
for k=numel(t):-1:2
    ind((tx>t{k-1}(nt(k-1)))&(tx<=t{k}(nt(k)))) = k;
end;

if (isa(Xt,'double'))
    Xt_loc = zeros(size(Xt,1),numel(tx));
else
    x = cell(d,numel(tx));
end;
for i=1:numel(tx)
    p = cheb2_interpolant(t{ind(i)},tx(i));
    if (isa(Xt,'double'))
        % in tt_tensor format, just concatenate all interpolators
        Xt_loc(:,i) = Xt(:, sum(nt(1:ind(i)-1))+1:sum(nt(1:ind(i))))*p.';
    else
        % given {d,R} storage, return a similar {d,numel(t_needed)}
        x(:,i) = X(1:d, ind(i));
        [rd,nd,md,rdp]=size(x{d,i});
        x{d,i} = reshape(x{d,i}, rd*nd*md, rdp);
        x{d,i} = x{d,i}*(Xt{ind(i)}*p.');
        x{d,i} = reshape(x{d,i}, rd, nd, md);
    end;
end;

if (isa(Xt,'double'))
    % apply interps. all at once
    x = x*Xt_loc;
end;
end
