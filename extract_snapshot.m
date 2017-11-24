% Interpolation on a Chebyshev or uniform grid in time
%   function [x] = extract_snapshot(X,t,tx, scheme)
%
% Extracts x=X(:,...,:,tx) from tamen outputs X (solution TT tensor(s)) and 
% t (cell array of time grid points) via barycentric Chebyshev
% interpolation for the Chebyshev scheme (scheme='cheb'), 
% and linear interpolation for the Crank-Nicolson scheme (scheme='cn').
% To avoid extrapolation, tx should be in [0,1].
% If X is in {d+1,R} format, returns x in {d,numel(tx)} format.
% If X is a tt_tensor (d+1-dimensional), returns x as a d-dimensional
% tt_tensor, or a 1 x numel(tx) cell of such, if numel(tx)>1.

function [x]=extract_snapshot(X,t,tx, scheme)
if (nargin<4)||(isempty(scheme))
    scheme = 'cheb';
end;

% number of grid points in each subinterval
nt = cellfun(@numel, t);
% find which interval tx belongs to
ind = ones(numel(tx), 1);
for k=numel(t):-1:2
    ind((tx>t{k-1}(nt(k-1)))&(tx<=t{k}(nt(k)))) = k;
end;
% Construct interpolants
x = cell(1, numel(tx));
for i=1:numel(tx)
    % Parse different inputs
    if (isa(X, 'tt_tensor'))
        d = X.d-1;
        Xt = X{d+1};
        x{1,i} = chunk(X,1,d);
    elseif (isa(X{ind(i)}, 'tt_tensor'))
        d = X{ind(i)}.d-1;
        Xt = X{ind(i)};
        x{1,i} = chunk(Xt,1,d);
        Xt = Xt{d+1};
    else % X is a {d,R} cell
        d = size(X,1)-1;
        if (size(x,1)<d)
            x = [x; cell(d-size(x,1), numel(tx))];                     %#ok
        end;
        x(:,i) = X(1:d, ind(i));
        Xt = X{d+1, ind(i)};
    end;
    
    % Interpolate the time block
    if (~isempty(strfind(lower(scheme), 'cheb')))
        p = cheb2_interpolant(t{ind(i)},tx(i));
        Xt = Xt*p.';
    elseif (~isempty(strfind(lower(scheme), 'cn')))
        % Interp linearly
        k = find(tx(i)<=t{ind(i)}, 1)-1; % the point on the left of tx
        p = [tx(i)-t{ind(i)}(k)  t{ind(i)}(k+1)-tx(i)]/(t{ind(i)}(k+1)-t{ind(i)}(k));
        Xt = Xt(:,k)*p(2) + Xt(:,k+1)*p(1);
    else
        error('Only Chebyshev and CN (Crank-Nicolson) schemes are implemented so far');
    end;
    
    % Contract the time block with the spatial part
    if (isa(x{1,i}, 'tt_tensor'))
        x{1,i} = x{1,i}*Xt;
    else
        [rd,nd,md,rdp]=size(x{d,i});
        x{d,i} = reshape(x{d,i}, rd*nd*md, rdp);
        x{d,i} = x{d,i}*Xt;
        x{d,i} = reshape(x{d,i}, rd, nd, md);
    end;
end;

if (numel(tx)==1)&&(isa(x{1}, 'tt_tensor'))
    x = x{1};
end;
end
