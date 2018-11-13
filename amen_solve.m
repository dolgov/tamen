% Alternating Minimal Energy algorithm in the TT format for linear systems
%   function [x,opts,swp] = amen_solve(A,y,tol,opts,x0,aux)
%
% Tries to solve the linear system Ax=y using the the AMEn iteration.
% 
% A is a square matrix in the TT format (for good performance it should
% also be nonnegative definite, but can be non-symmetric). It can be given as 
% either a tt_matrix class from the TT-Toolbox, or a cell array of size d x R,
% containing TT cores (see help ttdR).
%
% y is the right-hand side in the TT format. Can be given as either a
% tt_tensor class from the TT-Toolbox, or a cell array of size d x R, see
% help ttdR. The solution x is returned in the same format as y, but if it
% is {d,R}, R is always 1.
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
% The third output is the actual number of full sweeps carried out (<=nswp)
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
% Please see the reference:
%       S. Dolgov, D. Savostyanov,
%       http://epubs.siam.org/doi/10.1137/140953289
% for more description. 
% Feedback may be sent to {sergey.v.dolgov,dmitry.savostyanov}@gmail.com
%
% See also
%   TT-Toolbox: http://github.com/oseledets/TT-Toolbox
%   tAMEn:
%       S. Dolgov, https://doi.org/10.1515/cmam-2018-0023
%               or http://arxiv.org/abs/1403.8085

function [x,opts,swp]=amen_solve(A,y,tol,opts,x0,aux)

% Parse the right-hand side
[d,n,~,~,vectype]=grumble_vector(y,'y');
if (isa(y, 'tt_tensor'))
    y = core2cell(tt_matrix(y, n, 1));
end;
% Check the matrix for consistency
grumble_matrix(A,'A',d,n);
if (isa(A, 'tt_matrix'))
    A = core2cell(A);
end;
% Parse the initial guess
if ((nargin<5)||(isempty(x0)))
    % No x0 given, initialize as random rank-2
    x = cell(d,1);
    rx = [1;2*ones(d-1,1);1];
    for i=d:-1:2
        x{i} = randn(rx(i), n(i), 1, rx(i+1));
        [~,x{i},rx(i)] = orthogonalise_block([],x{i},-1);
    end;
    x{1} = randn(1, n(1), 1, rx(2));
else
    x = x0;
    % Check it for consistency
    grumble_vector(x,'x',d,n);
    if (isa(x, 'tt_tensor'))
        x = core2cell(tt_matrix(x,n,1));
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
                aux_in(:,i) = core2cell(tt_matrix(aux{i},n,1));
            else
                error('All aux vectors must be either tt_tensors or {d,R}s');
            end;
        end;
        aux = aux_in;
    else
        % Aux contains {d,R}, check for consistency
        grumble_vector(aux,'aux',d,n);
    end;
else
    aux = [];
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
if (opts.kickrank==0)
    rz = zeros(d+1,1);
end;
for i=d:-1:2
    z{i} = randn(rz(i), n(i), 1, rz(i+1));
    [~,z{i},rz(i)] = orthogonalise_block([],z{i},-1);
end;
z{1} = randn(1, n(1), 1, rz(2));
ZAX = [];
ZY = [];

for swp=1:opts.nswp
    % Run the AMEn solver
    [x,rx,z,ZAX,ZY,opts,errs,resids]=amenany_sweep(x,A,y,z, tol, opts, ZAX, ZY, aux);
    
    % Check and report error levels
    max_err = max(errs);
    max_res = max(resids);
    if (opts.verb>0)
        fprintf('amen_solve: swp=%d, err=%3.3e, res=%3.3e, rank=%d\n', swp, max_err, max_res, max(rx));
    end;    
    % Check the stopping criteria
    if ((strcmp(opts.trunc_norm, 'fro'))&&(max_err<tol))||((~strcmp(opts.trunc_norm, 'fro'))&&(max_res<tol))
        break;
    end;
end;

% Cast spatial solution to the desired form
if (strcmp(vectype, 'tt_tensor'))
    x = cell2core(tt_matrix, x);
    x = x.tt;
end;
end

