% Time-dependent Alternating Minimal Energy algorithm in the TT format
%   function [X,t,opts,x] = tamen(X,A,tol,opts,obs)
%
% Tries to solve the linear ODE dx/dt=Ax in the Tensor Train format using 
% the AMEn iteration and the Chebyshev or Crank-Nicolson discretization 
% of the time derivative. The time interval is [0,1], please rescale A 
% appropriately. If necessary, the scheme splits the interval adaptively. 
% This may result in different forms of output, depending on how many inner
% splittings were necessary, see below.
%
% X is the whole space-time solution, which must have the form of the 
% tensor train, where the last TT block should carry the time variable.
% X can be either a tt_tensor class from the TT-Toolbox, 
% or a cell array of size d+1 x 1, containing TT cores (see help ttdR).
% In the latter case, output can be a d+1 x R cell array if the time interval 
% was split into R subintervals during refinement of time discretization. 
% For convenience of subsequent invocations, the input can also have a
% d+1 x R form with R>1. Only the latest column (:,R) will be extracted,
% since it is the column where the last snapshot from the previous run is
% located.
% If X was given as tt_tensor and the time interval was split, the output 
% will be a 1 x R cell of tt_tensors.
% 
% The initial state is extracted as the last snapshot from X. Therefore,
% in the first time step you can pass it in the form X = tkron(x0, tt_ones(nt)) 
% or similar, such that X(:,...,:,nt)=x0. In the subsequent steps it is
% sufficient to just pass X from the previous step.
%
% A must be a square nonpositive definite matrix in the TT format, either a
% tt_matrix class from the TT-Toolbox, or a cell array of size D x R,
% containing TT cores (see help ttdR), or a function, constructing A in
% either of the previous two formats. A can be either stationary, in this
% case D==d, and A contains the spatial part only, or time-dependent, in
% this case D==d+1, the first d blocks contain the spatial part, and the
% last block must be of size r x nt x nt, diagonal w.r.t. the nt x nt part.
% The whole matrix is thus diagonal w.r.t. the time, and the diagonal 
% contains the spatial matrices evaluated at the time grid points, A(t_j)
%
% If A is a function, it should take two arguments X,t: X is the solution as
% described above (might be useful if the problem is nonlinear), 
% and t is the vector of time grid points on the current time interval.
%
% tol is the relative tensor truncation and stopping threshold for the
% AMEn method. The error may be measured either in the frobenius norm, or
% as the residual. See opts.trunc_norm parameter below.
%
% opts is a struct containing tuning parameters. For the description of
% AMEn-only parameters please see help amen_solve. The tAMEn related fields
% are the following.
%   nswp (default 20):          maximal number of AMEn sweeps
%   kickrank (default 4):       TT rank of the residual/enrichment in AMEn
%   local_iters (default 100):  Maximal number of bicgstab iterations for
%                               local problems
%   trunc_norm (default 'fro'): Norm to measure the error for truncation
%                               and stopping purposes. Can be either 'fro'
%                               (Frobenius norm) or 'res' (residual in the
%                               local system)
%   time_error_damp (def. 100): A time step is rejected if the time
%                               discretization error is greater than
%                               tolerance, divided by time_error_damp
%   time_scheme (def. 'cheb'):  Time discretization scheme. Can be either
%                               'cheb' (Chebyshev differentiation) or
%                               'cn' (Crank-Nicolson).
%   verb (default 1):           verbosity level: silent (0), sweep info (1)
%                               or full info for each block (2)
% opts is returned in outputs, and may be reused in the forthcoming calls.
%
% obs is an optional parameter containing generating vectors of linear
% invariants, if necessary. If the exact system conserves the quantities
% dot(c_m,x), and c_m are representable in the TT format, this property may
% be ensured for tAMEn by passing in obs either a 1 x M cell array, with
% obs{m}=c_m in the tt_tensor class from the TT-Toolbox, or a cell array of
% size d x M containing TT cores (see help ttdR).
%
% 
% The outputs are X (as described above), a cell array t of vectors of 
% time discretization points in (0,1], opts (with missed fields now populated 
% with their default values), and the last snapshot x=X(:,...,:,end).
% Every cell in t corresponds to a particular subinterval of the adaptive
% scheme. Use cell2mat(t) to obtain the global vector of points.
% The solution can be interpolated to any time in [0,1] using 
% extract_snapshot routine.
%
%
% ******************
% Please see the reference:
%   S. Dolgov, http://arxiv.org/abs/1403.8085
% for more description. 
% Feedback may be sent to sergey.v.dolgov@gmail.com
%
% See also
%   TT-Toolbox: http://github.com/oseledets/TT-Toolbox
%   amen_solve: AMEn algorithm: 
%       S. Dolgov, D. Savostyanov,
%       http://epubs.siam.org/doi/10.1137/140953289

function [X_global,t_global,opts,x] = tamen(X,A,tol,opts,obs)
% Parse the solution
[d,n,~,rx,vectype]=grumble_vector(X,'x');
d = d-1; % the last block corresponds to time, distinguish it from "space"
% by treating the tensor as d+1-dimensional.
if (isa(X, 'cell'))&&(size(X,2)>1)
    X = X(:,end);
end;
if (isa(X{1}, 'tt_tensor'))
    X = X{1};
end;
if (isa(X, 'tt_tensor'))
    % Extract tt_tensor content
    X = core2cell(X);
    for i=1:d
        X{i} = reshape(X{i}, rx(i), n(i), 1, rx(i+1));
    end;
end;

% Parse the options
if (nargin<4)||(isempty(opts))
    opts = struct;
end;
% Parse opts parameters. We just populate what we do not have by defaults
if (~isfield(opts, 'nswp'));           opts.nswp=20;              end;
if (~isfield(opts, 'kickrank'));       opts.kickrank=4;           end;
if (~isfield(opts, 'verb'));           opts.verb=1;               end;
if (~isfield(opts, 'local_iters'));    opts.local_iters=100;      end;
if (~isfield(opts, 'trunc_norm'));     opts.trunc_norm='fro';     end;
if (~isfield(opts, 'time_error_damp'));opts.time_error_damp=100;  end;
if (~isfield(opts, 'time_scheme'));    opts.time_scheme='cheb';   end;

% The local_iters parameter affects only the spatial blocks. The temporal
% system will be solved directly here, so just exit from the inner iteration
if (numel(opts.local_iters)==1)
    opts.local_iters = [opts.local_iters*ones(d,1); 0];
end;

% Orthogonalize the spatial part of the solution
for i=1:d-1
    [X{i+1},X{i},rx(i+1)] = orthogonalise_block(X{i+1},X{i},1);
end;

% Parse auxiliary enrichments
if ((nargin>=5)&&(~isempty(obs)))
    if (~isa(obs, 'cell'))
        error('Aux vectors must be given in a cell array');
    end;
    Raux = size(obs,2);
    Aux = cell(d,Raux);
    if (isa(obs{1}, 'tt_tensor'))
        % Aux contains tt_tensors
        raux = ones(d+1,Raux);
        for i=1:Raux
            if (isa(obs{i}, 'tt_tensor'))
                [~,~,~,raux(:,i)]=grumble_vector(obs{i},'aux',d,n(1:d));
                Aux(1:d,i) = core2cell(tt_matrix(obs{i}, n(1:d), 1));
            else
                error('All aux vectors must be either tt_tensors or {d,R}s');
            end;
        end;
    else
        % Aux contains {d,R}
        [~,~,~,raux]=grumble_vector(obs,'aux',d,n(1:d));
        Aux(1:d,:) = obs;
    end;
else
    Aux = [];
    raux = [];
    Raux = 0;
end;

% Prepare a random initial guess for z
z = cell(d+1,1);
if (isscalar(opts.kickrank))
    rz = [1;opts.kickrank*ones(d,1);1];
elseif (numel(opts.kickrank)==d)
    rz = [1;opts.kickrank;1]; % we might prefer a particular rank shape
else
    error('number of different rz components should be %d', d);
end;
for i=d+1:-1:2
    z{i} = randn(rz(i), n(i), 1, rz(i+1));
    [~,z{i},rz(i)] = orthogonalise_block([],z{i},-1);
end;
z{1} = randn(1, n(1), 1, rz(2));

% Storage for all time steps
X_global = cell(d+1,10);
t_global = cell(10,1);

if (~isempty(strfind(lower(opts.time_scheme), 'cheb')))
    % Spectral differentiators
    [t_coarse,St]=chebdiff(n(d+1));
    % Temporal RHS
    rhst = sum(St, 2);
    rhst = reshape(rhst, 1, n(d+1));
    iSt = inv(St);
    % Twice as many points for grid refinement
    [t_fine,St2]=chebdiff(n(d+1)*2);
    % Temporal RHS
    % Construct interpolants
    P = cheb2_interpolant(t_coarse, t_fine);
    iSt2 = St2\P;
elseif (~isempty(strfind(lower(opts.time_scheme), 'cn')))
    if (mod(n(d+1),2)==0)
        error('Crank-Nicolson scheme requires odd number of time points');
    end;
    % Crank-Nicolson matrices
    Mt = spdiags(ones(n(d+1),1)*([0.5 0.5]/(n(d+1)-1)), -1:0, n(d+1), n(d+1));
    Mt(1,1) = 0; % initial state is included into the storage
    Gt = spdiags(ones(n(d+1),1)*[-1 1], -1:0, n(d+1), n(d+1));
    rhst = eye(1,n(d+1));
    t_coarse = (0:(n(d+1)-1))'/(n(d+1)-1);
    Mt2 = spdiags(ones(2*n(d+1)-1,1)*([0.25 0.25]/(n(d+1)-1)), -1:0, 2*n(d+1)-1, 2*n(d+1)-1);
    Mt2(1,1) = 0; % initial state is included into the storage
    Gt2 = spdiags(ones(2*n(d+1)-1,1)*[-1 1], -1:0, 2*n(d+1)-1, 2*n(d+1)-1);
else
    error('Only Chebyshev and CN (Crank-Nicolson) schemes are implemented so far');
end;

% Time stepping
time_step = 1;
time_global = 0;
t_cnt = 0;
while (time_global<1)
    % Don't advance beyond t=1
    if (time_global + time_step>1)
        time_step = 1 - time_global;
    end;
    % Parse the matrix and compute it in all possible ways
    [As,Ra,ras] = parse_matrix(A,n,X, time_global + t_coarse*time_step, vectype);
    for k=1:Ra
        As{1,k}=As{1,k}*time_step;
    end;
    if (~isempty(strfind(lower(opts.time_scheme), 'cheb')))
        % Put the Chebyshev temporal derivative into the storage
        As{d+1,Ra+1} = reshape(St, 1, n(d+1), n(d+1));        
    elseif (~isempty(strfind(lower(opts.time_scheme), 'cn')))
        % Crank-Nicolson matrices
        As_true = As(d+1,1:Ra); % save this for residual estimation below
        for k=1:Ra
            % Multiply by mass
            As{d+1,k} = reshape(As{d+1,k}, ras(d+1,k), n(d+1)*n(d+1));
            As{d+1,k} = As{d+1,k}.';
            As{d+1,k} = reshape(As{d+1,k}, n(d+1), n(d+1)*ras(d+1,k));
            As{d+1,k} = Mt*As{d+1,k};
            As{d+1,k} = reshape(As{d+1,k}, n(d+1)*n(d+1), ras(d+1,k));
            As{d+1,k} = As{d+1,k}.';
            As{d+1,k} = reshape(As{d+1,k}, ras(d+1,k)*n(d+1), n(d+1));
        end;
        As{d+1,Ra+1} = Gt;
    end;
    
    % Save the original X for if we need to reject the time step   
    rx = [cellfun(@(x)size(x,1), X); 1];
    X_initial = X;
    % Extract a precursor for x0, and also the spatial RHS
    x0 = X;
    x0{d} = reshape(x0{d}, rx(d)*n(d), rx(d+1));
    % Now the last snapshot of the previous call is our x0
    x0{d} = x0{d}*X{d+1}(:,end); % Now x0 is ready, and |x0|=|x0{d}|.
    x0{d} = reshape(x0{d}, rx(d), n(d), 1);
    % x0's second norm
    x0norm = norm(x0{d}, 'fro');
    % Temporal RHS
    x0{d+1} = rhst;
        
    % Drop residual reductions, since x0 has changed
    ZAX = []; ZY = [];
    
    % Start amen sweeps
    for swp=1:opts.nswp
        % Run the spatial solver                                      
        [X,rx,z,ZAX,ZY,opts,errs,resids,XAX,XY,XAUX]=amenany_sweep(X, As, x0, z, tol, opts, ZAX, ZY, Aux);
        % Check and report error levels
        max_err = max(errs);
        max_res = max(resids);
        
        % The last temporal system must be solved directly.
        u0 = XY{d+1}; % size rxs(d+1) x 1
        % Correct the 2nd norm. It is transparent: we adjust the norm of the
        % projected initial guess, not the final solution. That is, it works
        % also if the matrix does not conserve the second norm.
        if (Raux>0)
            % AUX'*x0 must be conserved, rescale only the orth. complement
            Caux = zeros(rx(d+1), sum(raux(d+1,:)));
            pos = 1;
            for k=1:Raux
                Caux(:,pos:pos+raux(d+1,k)-1) = XAUX{k,d+1}; % each xaux is rxs(d+1) x 1
                pos = pos+raux(d+1,k);
            end;
            [Caux,~]=qr(Caux,0);
            u0_save = Caux'*u0;
            u0_change = u0 - Caux*u0_save;
            u0_change = u0_change*sqrt(abs(x0norm^2 - norm(u0_save)^2))/norm(u0_change);
            u0 = Caux*u0_save + u0_change;
        else
            u0 = u0*(x0norm/norm(u0)); % this can perturb the solution if the time discretization is insufficient
        end;
        if (~isempty(strfind(lower(opts.time_scheme), 'cheb'))) 
            % Chebyshev reduced ODE solver
            % Spatial matrix is reduced to XAX{d+1} [rxs(d+1) x rxs(d+1) x ras(d+1)]
            At = zeros(rx(d+1)*n(d+1), rx(d+1)*n(d+1));
            % Assemble the space-time systems
            for k=1:Ra
                Atk = reshape(XAX{k,d+1}, rx(d+1)*rx(d+1), ras(d+1,k));
                Atk2 = reshape(As{d+1,k}, ras(d+1,k), n(d+1)*n(d+1));
                Atk2 = Atk*Atk2;
                Atk2 = reshape(Atk2, rx(d+1), rx(d+1), n(d+1), n(d+1));
                Atk2 = permute(Atk2, [1,3,2,4]);
                Atk2 = reshape(Atk2, rx(d+1)*n(d+1), rx(d+1)*n(d+1));
                At = At+Atk2;
            end;
            % Solve the system with the square matrix first
            AT = eye(rx(d+1)*n(d+1)) + kron(iSt, eye(rx(d+1)))*At;
            yt = repmat(u0, 1, n(d+1));
            yt = reshape(yt, rx(d+1)*n(d+1), 1);
            xt = AT\yt;
            X{d+1} = reshape(xt, rx(d+1), n(d+1));
            
            % We want to estimate the residual by using a rectangular Cheb matrix
            AT2 = kron(P, eye(rx(d+1))) + kron(iSt2,eye(rx(d+1)))*At;
            yt = repmat(u0, 1, 2*n(d+1));
            yt = reshape(yt, rx(d+1)*n(d+1)*2, 1);
            time_resid = norm(AT2*xt - yt, 'fro')/norm(yt, 'fro');
        elseif (~isempty(strfind(lower(opts.time_scheme), 'cn'))) 
            % Crank-Nicolson reduced ODE solver
            At = sparse(rx(d+1)*n(d+1), rx(d+1)*n(d+1));
            for k=1:Ra
                At2 = As_true{k};
                if (ras(d+1,k)>1)
                    At2 = reshape(At2, ras(d+1,k), n(d+1)*n(d+1));
                    for j=1:ras(d+1,k)
                        At = At + kron(reshape(At2(j,:), n(d+1), n(d+1)), sparse(XAX{k,d+1}(:,:,j)));
                    end;
                else
                    At = At + kron(At2, sparse(XAX{k,d+1}));
                end;
            end;
            AT = kron(Gt, speye(rx(d+1))) + kron(Mt, speye(rx(d+1)))*At;
            yt = [u0; zeros(rx(d+1)*(n(d+1)-1), 1)];
            xt = AT\yt;
            X{d+1} = reshape(xt, rx(d+1), n(d+1));
            
            % estimate the residual
            % interpolate Ax
            xt2 = At*xt;
            xt2 = reshape(xt2, rx(d+1), n(d+1));
            yt2 = zeros(rx(d+1), 2*n(d+1)-1);
            yt2(:, 1:2:(2*n(d+1)-1)) = xt2;
            yt2(:, 2:2:(2*n(d+1)-2)) = (xt2(:,1:(n(d+1)-1))+xt2(:,2:n(d+1)))*0.5;
            yt2 = yt2*Mt2.';
            yt2 = yt2/Gt2.';
            % interpolate x
            xt2 = zeros(rx(d+1), 2*n(d+1)-1);
            xt = reshape(xt, rx(d+1), n(d+1));
            xt2(:, 1:2:(2*n(d+1)-1)) = xt;
            xt2(:, 2:2:(2*n(d+1)-2)) = (xt(:,1:(n(d+1)-1))+xt(:,2:n(d+1)))*0.5;
            time_resid = norm(xt2 + yt2 - u0*ones(1,2*n(d+1)-1), 'fro')/norm(u0)/sqrt(2*n(d+1)-1);
        end;
        
        if (opts.verb>0)
            fprintf('tamen: t=%g, swp=%d, err=%3.3e, res=%3.3e, time_res=%3.3e, rank=%d\n', time_global+time_step, swp, max_err, max_res, time_resid, max(rx));
        end;
        
        % A ratio of the time error and the tolerance
        dtime_step = tol/(opts.time_error_damp*time_resid);
        reject = false;
        if (dtime_step<1)&&(swp>1)
            % If the discretization is severely insufficient, reject this step
            X = X_initial;
            reject = true;
            break;
        end;
        
        % Check AMEn convergence
        if (swp>=opts.nswp)||((strcmp(opts.trunc_norm, 'fro'))&&(max_err<tol))||((~strcmp(opts.trunc_norm, 'fro'))&&(max_res<tol))
            % The current time step converged, include it and advance time
            break;
        end;
    end;

    if (opts.verb>0)
        fprintf('\t t=%g, time_res=%3.3e, rank=%d, ', time_global+time_step, time_resid, max(rx));
    end;
    if (~reject)
        % this time step was ok, register it
        t_cnt = t_cnt + 1;
        if (t_cnt>size(X_global,2))
            % allocate more cells
            X_global = [X_global, cell(d+1,10)];                       %#ok
            t_global = [t_global; cell(10,1)];                         %#ok
        end;
        X_global(:,t_cnt) = X;
        t_global{t_cnt} = time_global + t_coarse*time_step;
        time_global = time_global + time_step;
    end;
    % Amend the time step to the current error
    if (~isempty(strfind(lower(opts.time_scheme), 'cheb')))
        time_step = time_step*min(0.5*(dtime_step)^(1/n(d+1)), 2);
    elseif (~isempty(strfind(lower(opts.time_scheme), 'cn')))
        time_step = time_step*min(0.5*sqrt(dtime_step), 2); % Crank-Nicolson is 2nd order
    end;
    if (opts.verb>0)
        fprintf('new time step=%3.3e ', time_step);
        if (reject)
            fprintf('\t !REJECT!\n\n');
        else
            fprintf('\n\n');
        end;
    end;
end;

X_global = X_global(:,1:t_cnt);
t_global = t_global(1:t_cnt,:);


% Cast the whole solution to tt_tensor if we want
if (strcmp(vectype, 'tt_tensor'))
    for k=1:t_cnt
        X_global{1,k} = cell2core(tt_matrix, X_global(:,k));
        X_global{1,k} = X_global{1,k}.tt;
    end;
    X_global = X_global(1,:);
    if (t_cnt==1)
        X_global = X_global{1};
    end;
end;

% Return the last snapshot
if (nargout>3)
    Xt = X{d+1};
    Xt = Xt(:,end);
    x = X(1:d);
    x{d} = reshape(x{d}, rx(d)*n(d), rx(d+1));
    x{d} = x{d}*Xt;
    x{d} = reshape(x{d}, rx(d), n(d));
    rx(d+1) = 1;
    for i=1:d
        x{i} = reshape(x{i}, rx(i), n(i), 1, rx(i+1)); % store the sizes in
    end;
    if (strcmp(vectype, 'tt_tensor'))
        x = cell2core(tt_matrix,x);
        x = x.tt;
    end;
end;

% local_iters is normally a scalar, return it
opts.local_iters = opts.local_iters(1);
end


function [As,Ra,ras]=parse_matrix(A,n,X,t,vectype)
% Parse the matrix. It will be harder than in amen_solve, since we may have
% either stationary or time-dependent matrix, or a function generating
% either of those

% space-only dimension
d = numel(n)-1;

if ((~isa(A, 'cell'))&&(~isa(A, 'tt_matrix')))
    % This is a function and takes (X,t)
    % Cast spatial solution to the desired form    
    if (strcmp(vectype, 'tt_tensor'))
        for i=1:d+1
            r1 = size(X{i},1);
            r2 = size(X{i},4);
            X{i} = reshape(X{i}, r1, n(i), r2);
        end;        
        X = cell2core(tt_tensor, X);
        % Otherwise just cast A on it
    end;
    A = A(X,t);
end;

% Then, get the dimension of the matrix
if (isa(A, 'tt_matrix'))
    D = A.d;
else
    D = size(A,1);
end;
% The only allowed dimensions are d and d+1, where d is the dimension of
% the space
switch(D)
    case d
        % Matrix is stationary
        [~,~,Ra,ras]=grumble_matrix(A,'A',d,n(1:d));
        ras = [ras, ones(d+1,1)]; % this is needed for the time derivative
        ras = [ras; ones(1,Ra+1)];
    case d+1
        % Matrix is time-dependent
        [~,~,Ra,ras]=grumble_matrix(A,'A',d+1,n);
        ras = [ras, ones(d+2,1)]; % for time derivative in further computations
    otherwise
        error('dim(A) must be either d (if A is stationary) or d+1 (time-dependent)');
end;
% Now copy the cores
As = cell(d+1, Ra+1);
if (isa(A, 'tt_matrix'))
    As(1:D,1) = core2cell(A);
else
    As(1:D,1:Ra) = A;
end;
% Populate the temporal (:,R+1) parts with identities
for i=1:d
    sparseflag = true;
    for k=1:Ra
        if (~issparse(As{i,k})); sparseflag=false; break; end;
    end;
    if (sparseflag)
        % The sparse A-block was introduced for something. Such as large n(i). 
        % Use a sparse identity also.
        As{i,Ra+1} = speye(n(i));
    else
        As{i,Ra+1} = reshape(eye(n(i)), 1, n(i), n(i));
    end;
end;
% Fill temporal (d+1,:) matrix parts with identities if necessary
for k=1:Ra
    if (isempty(As{d+1,k}))
        As{d+1,k} = eye(n(d+1));
    end;
    As{d+1,k} = reshape(As{d+1,k}, ras(d+1,k)*n(d+1), n(d+1));
    As{d+1,k} = sparse(As{d+1,k});
    As{d+1,k} = -As{d+1,k};
end;
end

