% Barycentric interpolation from the Chebyshev kind-2 points x 
% (used in tamen, inverse CDF, etc.) to an arbitrary set y.
%   function [p,ind]=cheb2_interpolant(x, y, nloc)
%
% The basis points x are assumed to be sorted.
% Barycentric weights are 
% w_j = (-1)^j*delta_j, where delta_j=1/2, j=N, and 1, otherwise
% Barycentric interpolation is then
% p(y) = w_j/(y-t_j) / Z, where Z = sum_j w_j/(y-t_j),
% see [J-P. Berrut and L. N. Trefethen, SIAM REVIEW V. 46, No. 3, P. 501].
%
% This procedure allows composite piecewise Chebyshev grids. In this case,
% nloc specifies the size of a single set of points. Interpolator p can be
% returned in two ways: if only p is present in the output, p is a sparse
% matrix of size numel(y) x numel(x). If additionally ind is present, p is
% a dense matrix of size numel(y) x nloc (acting on a single segment), and
% ind is an integer numel(y) x nloc matrix, such that x(ind(i,:)) is the
% segment of x required for the i-th point in y.

function [p,ind]=cheb2_interpolant(x, y, nloc)

n = numel(x);
M = numel(y);
% Exclude trivial (but ill-defined for w) case
if (n==1)
    p = ones(M,1);
    return;
end;

if (nargin<3)||(isempty(nloc))
    nloc = n;
end;
n_sets = n/nloc;

% weights
w = (-1).^((nloc-1):-1:0);
w(nloc) = w(nloc)*0.5;
w = repmat(w, 1, n_sets);
% w should be multiplied by (x0-xj), since we don't have x0. Compute x0 by
% relating the given points to the reference Chebyshev points on [-1,1].
j0 = floor(nloc/2); % make the cos smaller for stability
x0 = (2*x(nloc-j0 + (0:n_sets-1)*nloc) - x(nloc + (0:n_sets-1)*nloc)*(1+cos(pi*j0/nloc)))/(1 - cos(pi*j0/nloc));
x0 = reshape(x0, 1, n_sets);
x0 = repmat(x0, nloc, 1); % Equalize the size: expand x0 within each interval
x0 = reshape(x0, 1, n);

x_test = reshape(x, 1, n);
w = w.*(x0-x_test); % correct w for the absent x0, individually on each interval

% find appropriate interval for each y
x_test = x(nloc:nloc:n); % size 1 x n_sets
x_test = reshape(x_test, 1, n_sets);
y_test = reshape(y, M, 1);
y_test = repmat(y_test, 1, n_sets);
x_test = repmat(x_test, M, 1);
ind = sum(y_test>x_test, 2); % implement find over 2nd dim via the sum of logicals
% some of the y might be larger than the last x
ind(ind>=n_sets) = n_sets - 1;
% ind runs from 0 to n_sets-1
% expand to each interval constantly
ind = ind*nloc;
ind = repmat(ind, 1, nloc);
ind = ind + repmat(1:nloc, M, 1);
x_test = x(ind);
x_test = reshape(x_test, M, nloc); % These are the points surrounding each y
w = w(ind);
w = reshape(w, M, nloc);
y_test = reshape(y, M, 1);
y_test = repmat(y_test, 1, nloc);

% Barycentric formula
p = w./(y_test - x_test);
Z = sum(p,2); % M x 1
Z = repmat(Z, 1, nloc);
p = p./Z;
p(isnan(p))=1;
% p is only of size M x nloc, defined on each individual interval
% We need to expand it by zeros to full n
if (n_sets>1)&&(nargin==1)
    ind_rows = repmat((1:M)',1,nloc);
    p = sparse(ind_rows, ind, p, M, n);
end;
end
