% Alternating Minimal Energy algorithm in the TT format for linear systems
%   function [x,opts]=amen_solve(A,y,tol,opts,x0,aux)
%
% Tries to solve the linear system Ax=y using the the AMEn iteration.
% 
% A is the square matrix in the TT format (for good performance it should
% also be nonnegative definite, but can be non-symmetric). It can be given as 
% either a tt_matrix class from the TT-Toolbox, or a cell array of size d x R,
% containing TT cores (see help ttdR).
%
% y is the right-hand side in the TT format. Can be given as either a
% tt_tensor class from the TT-Toolbox, or a cell array of size d x R, see
% help ttdR. The solution x is returned in the same format as y.
%
% tol is the relative tensor truncation and stopping threshold.
% The error may be measured either in the frobenius norm, or
% as the residual. See opts.trunc_norm parameter below.
%
% opts is a struct containing the following parameters for fine tuning of
% the algorithm.
%   nswp (default 20):          Maximal number of AMEn sweeps
%                        ----------
%   kickrank (default 4):       TT rank of the residual/enrichment in AMEn
%   kickrank2 (default 0):      Rank for the secondary enrichment (for the
%                               residual itself) using random vectors.
%                               Nonzero values may give more reliability
%                               for very difficult systems.
%                        ----------
%   trunc_norm (default 'fro'): Norm to measure the error for truncation
%                               and stopping purposes. Can be either 'fro'
%                               (Frobenius norm) or 'res' (residual in the
%                               local system)
%   rmax (default Inf):         Maximal TT rank limit for the solution
%                        ----------
%   max_full_size (default 50): If the size of the local system is smaller
%                               than max_full_size, solve it directly,
%                               otherwise iteratively. Larger max_full_size
%                               gives higher accuracy, but may slow down
%                               the process.
%   local_iters (default 100):  Maximal number of bicgstab iterations for
%                               local problems
%   resid_damp (default 2):     Local accuracy gap. If the iterative solver
%                               is used for local problems, it's stopping
%                               tolerance is set to (tol/sqrt(d))/resid_damp
%   local_prec (default ''):    Set up the preconditioner for local
%                               problems. Under construction yet! Currently
%                               only 'r' (Right Block Jacobi) or '' (no
%                               preconditioner) are supported.
%                        ----------
%   verb (default 1):           Verbosity level: silent (0), sweep info (1)
%                               or full info for each block (2).
% opts is returned in outputs, and may be reused in the forthcoming calls.
%
% x0 is the initial guess in the TT format (either tt_tensor or {d,R}). If
% absent, a random tensor of TT ranks 2 is used.
%
% aux is the cell array containing additional vectors for the enrichment
% (optional). For example, if some of eigenvectors of A are known in the TT
% format, they may be passed in aux to improve the accuracy. If the vectors
% are passed as tt_tensor classes, aux must be of sizes 1 x R, each cell
% element must contain a tt_tensor. Alternatively, aux may be a {d,R} format 
%
%
% ******************
% Please see the references:
%       S. Dolgov, D. Savostyanov,
%       http://arxiv.org/abs/1301.6068  and
%       http://arxiv.org/abs/1304.1222
% for more description. 
% Feedback may be sent to {sergey.v.dolgov,dmitry.savostyanov}@gmail.com
%
% See also
%   TT-Toolbox: http://github.com/oseledets/TT-Toolbox
%   tAMEn:
%       S. Dolgov, http://arxiv.org/abs/1403.8085

function [x,opts]=amen_solve(A,y,tol,opts,x0,aux)

% Parse the right-hand side
[d,n,~,ry,vectype]=grumble_vector(y,'y');
if (isa(y, 'tt_tensor'))
    y = core2cell(y);
end;
% Parse the matrix
[~,~,~,ra]=grumble_matrix(A,'A',d,n);
if (isa(A, 'tt_matrix'))
    A = core2cell(A);
end;
% Parse the initial guess
if ((nargin<5)||(isempty(x0)))
    % No x0 given, initialize as random rank-2
    x = cell(d,1);
    rx = [1;2*ones(d-1,1);1];
    for i=d:-1:2
        x{i} = randn(n(i)*rx(i+1), rx(i));
        [x{i},~]=qr(x{i}, 0);
        rx(i) = size(x{i},2);
        x{i} = x{i}.';
    end;
    x{1} = randn(n(1), rx(2));
else
    x = x0;
    [~,~,~,rx]=grumble_vector(x,'x',d,n);
    if (isa(x, 'tt_tensor'))
        x = core2cell(x);
    end;
end;
% Parse auxiliary enrichments
if ((nargin>=6)&&(~isempty(aux)))
    if (~isa(aux, 'cell'))
        error('Aux vectors must be given in a cell array');
    end;
    Raux = size(aux,2);
    if (isa(aux{1}, 'tt_tensor'))
        % Aux contains tt_tensors
        raux = ones(d+1,Raux);
        aux_in = cell(d,Raux);
        for i=1:Raux
            if (isa(aux{i}, 'tt_tensor'))
                [~,~,~,raux(:,i)]=grumble_vector(aux{i},'aux',d,n);
                aux_in(:,i) = core2cell(aux{i});
            else
                error('All aux vectors must be either tt_tensors or {d,R}s');
            end;
        end;
        aux = aux_in;
    else
        % Aux contains {d,R}
        [~,~,~,raux]=grumble_vector(aux,'aux',d,n);
    end;
else
    aux = [];
    raux = [];
end;

% Parse opts parameters. We just populate what we do not have by defaults
if (nargin<4)||(isempty(opts))
    opts = struct;
end;
if (~isfield(opts, 'nswp'));           opts.nswp=20;              end;
if (~isfield(opts, 'kickrank'));       opts.kickrank=4;           end;
if (~isfield(opts, 'verb'));           opts.verb=1;               end;
if (~isfield(opts, 'trunc_norm'));     opts.trunc_norm='fro';     end;

% Prepare the initial guess for the residual
z = cell(d,1);
rz = [1;opts.kickrank*ones(d-1,1);1];
for i=d:-1:2
    z{i} = randn(n(i)*rz(i+1), rz(i));
    [z{i},~]=qr(z{i}, 0);
    rz(i) = size(z{i},2);
    z{i} = z{i}.';
end;
z{1} = randn(n(1), rz(2));
ZAX = [];
ZY = [];


for swp=1:opts.nswp
    % Run the AMEn solver
    [x,rx,z,rz,ZAX,ZY,opts,errs,resids]=amenany_sweep(n, x,rx,A,ra,y,ry,z,rz, tol, opts, ZAX, ZY, aux,raux);
    
    % Check and report error levels
    max_err = max(errs);
    max_res = max(resids);
    if (opts.verb>0)
        fprintf('amen_solve: swp=%d, err=%3.3e, res=%3.3e, rank=%d\n', swp, max_err, max_res, max(rx));
    end;    
    % Check the stopping criteria
    if (strcmp(opts.trunc_norm, 'fro'))
        if (max_err<tol); break; end;
    else
        if (max_res<tol); break; end;
    end;
end;

% Cast spatial solution to the desired form
if (strcmp(vectype, 'tt_tensor'))
    x = cell2core(tt_tensor, x);
else
    for i=1:d
        x{i} = reshape(x{i}, rx(i), n(i), 1, rx(i+1)); % store the sizes in
    end;
end;
end

% Check for consistency and retrieve the sizes of a vector in the TT format
function [d,n,Rx,rx,vectype]=grumble_vector(x,xname,d,n,Rx,rx)
if (isa(x, 'tt_tensor'))
    if (nargin>2)&&(~isempty(d))&&(x.d~=d)
        error('dim of %s is inconsistent',xname);
    else
        d = x.d;
    end;
    if (nargin>3)&&(~isempty(n))&&((numel(x.n)~=numel(n))||(~all(x.n==n)))
        error('mode sizes of %s are inconsistent', xname);
    else
        n = x.n;
    end;
    if (nargin>4)&&(~isempty(Rx))&&(Rx~=1)
        error('canonical rank of %s is inconsistent', xname);
    else
        Rx=1;
    end;
    if (nargin>5)&&(~isempty(rx))&&((numel(x.r)~=numel(rx))||(~all(x.r==rx)))
        error('TT ranks of %s are inconsistent', xname);
    else
        rx = x.r;
    end;
    vectype = 'tt_tensor';
else
    % {d,R} format
    if (nargin>2)&&(~isempty(d))&&(size(x,1)~=d)
        error('dim of %s is inconsistent',xname);
    else
        d = size(x,1);
    end;
    if (nargin>4)&&(~isempty(Rx))&&(Rx~=size(x,2))
        error('canonical rank of %s is inconsistent', xname);
    else
        Rx = size(x,2);
    end;
    if (strcmp(xname, 'x')||strcmp(xname, 'z'))&&(Rx>1)
        error('Tensor Chain format (R>1) is not allowed for input %s', xname);
    end;
    if (nargin<=3)||(isempty(n))
        n_in = ones(d,1);
    else
        if (numel(n)~=d)
            error('mode sizes of %s are inconsistent', xname);
        end;
    end;
    if (nargin<=5)||(isempty(rx))
        rx_in = ones(d+1,Rx);
    else
        if (size(rx,1)~=(d+1))||(size(rx,2)~=Rx)
            error('TT ranks of %s are inconsistent', xname);
        end;
    end;
    for k=1:Rx
        for i=1:d
            n_in(i) = size(x{i,k},2)*size(x{i,k},3);
            if (nargin>3)&&(~isempty(n))&&(n_in(i)~=n(i))
                error('mode size (%d,%d) of %s is inconsistent', i, k, xname);
            end;
            rx_in(i+1,k) = size(x{i,k},4);
            if (rx_in(i,k)~=size(x{i,k},1))
                error('TT rank (%d,%d) of %s is inconsistent', i, k, xname);
            end;
            if (nargin>5)&&(~isempty(rx))&&(rx_in(i+1,k)~=rx(i+1,k))
                error('TT rank (%d,%d) of %s is inconsistent', i, k, xname);
            end;
        end;
    end;
    n = n_in;
    rx = rx_in;
    vectype = 'dR';
end;
end

% Check for consistency and retrieve the sizes of a matrix in the TT format
function [d,n,Ra,ra]=grumble_matrix(a,aname,d,n,Ra,ra)
if (isa(a, 'tt_matrix'))
    if (nargin>2)&&(~isempty(d))&&(a.d~=d)
        error('dim of %s is inconsistent',aname);
    else
        d = a.d;
    end;
    if (nargin>3)&&(~isempty(n))&&((numel(a.n)~=numel(n))||(~all(a.n==n))||(~all(a.m==n)))
        error('mode sizes of %s are inconsistent', aname);
    else
        n = a.n;
    end;
    if (nargin>4)&&(~isempty(Ra))&&(Ra~=1)
        error('canonical rank of %s is inconsistent', aname);
    else
        Ra=1;
    end;
    if (nargin>5)&&(~isempty(ra))&&((numel(a.r)~=numel(ra))||(~all(a.r==ra)))
        error('TT ranks of %s are inconsistent', aname);
    else
        ra = a.r;
    end;
else
    % {d,R} format
    if (nargin>2)&&(~isempty(d))&&(size(a,1)~=d)
        error('dim of %s is inconsistent',aname);
    else
        d = size(a,1);
    end;
    if (nargin>4)&&(~isempty(Ra))&&(Ra~=size(a,2))
        error('canonical rank of %s is inconsistent', aname);
    else
        Ra = size(a,2);
    end;
    if (nargin<=3)||(isempty(n))
        n_in = ones(d,1);
    else
        if (numel(n)~=d)
            error('mode sizes of %s are inconsistent', aname);
        end;
    end;
    if (nargin<=5)||(isempty(ra))
        ra_in = ones(d+1,Ra);
    else
        if (size(ra,1)~=(d+1))||(size(ra,2)~=Ra)
            error('TT ranks of %s are inconsistent', aname);
        end;
    end;
    for k=1:Ra
        for i=1:d
            if (issparse(a{i,k}))
                % Sparse block, the TT ranks are set to ones, and nothing
                % more is possible now
                n_in(i) = size(a{i,k},1);
                if (size(a{i,k},2)~=n_in(i)) % Square matrix only for now
                    error('mode size (%d,%d) of %s is inconsistent', i, k, aname);
                end;
                if (nargin>3)&&(~isempty(n))&&(n_in(i)~=n(i))
                    error('mode size (%d,%d) of %s is inconsistent', i, k, aname);
                end;
                if (nargin>5)&&(~isempty(ra))&&(ra(i+1,k)~=1)
                    error('TT rank (%d,%d) of %s is inconsistent', i, k, aname);
                end;                
            else
                % Dense 4-dimensional blocks
                n_in(i) = size(a{i,k},2);
                if (size(a{i,k},3)~=n_in(i)) % Square matrix only for now
                    error('mode size (%d,%d) of %s is inconsistent', i, k, aname);
                end;
                if (nargin>3)&&(~isempty(n))&&(n_in(i)~=n(i))
                    error('mode size (%d,%d) of %s is inconsistent', i, k, aname);
                end;
                ra_in(i+1,k) = size(a{i,k},4);
                if (ra_in(i,k)~=size(a{i,k},1))
                    error('TT rank (%d,%d) of %s is inconsistent', i, k, aname);
                end;
                if (nargin>5)&&(~isempty(ra))&&(ra_in(i+1,k)~=ra(i+1,k))
                    error('TT rank (%d,%d) of %s is inconsistent', i, k, aname);
                end;
            end;
        end;
    end;
    n = n_in;
    ra = ra_in;
end;
end
