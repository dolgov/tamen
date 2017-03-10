% Time-dependent Alternating Minimal Energy algorithm in the TT format
%   function [X,opts,t,x] = tamen(X,A,tol,opts,obs)
%
% Tries to solve the linear ODE dx/dt=Ax in the Tensor Train format using 
% the AMEn iteration for the Chebyshev discretization of the time derivative.
% The time interval is [0,1], please rescale A if necessary.
% Nevertheless, the inner splitting and discretization of this interval is
% performed adaptively. This may result in different forms of output,
% depending on how many inner splittings were necessary, see below.
%
% X is the whole solution, which must have the form of the tensor train,
% where the last TT block should carry the time variable.
% X can be either a tt_tensor class from the TT-Toolbox, 
% or a cell array of size d+1 x 1, containing TT cores (see help ttdR).
% In the latter case, output can be a d+1 x R cell array if the time interval 
% was split into R subintervals during refinement of time discretization. 
% For convenience of subsequent invocations, the input can also have a
% d+1 x R form with R>1. Only the latest column (:,R) will be extracted,
% since it is the column where the last snapshot from the previous run is
% located.
% If X was given as tt_tensor and many time intervals were necessary, the
% output will still be a single tt_tensor, obtained by direct summation of
% tt_tensors from different subintervals. !!!NOTE!!! This may consume
% a lot of memory, since no truncation is applied. Consider using the {d,R}
% format instead if you expect your matrix to be stiff.
% 
% The initial state is extracted as the last snapshot from X. Therefore,
% in the first time step you can pass in the form X = tkron(x0, tt_ones(nt)) 
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
% contains the spatial matrices evaluated at the Chebyshev points, A(t_j)
%
% If A is a function, it should take two arguments X,t: X is the solution as
% described above (might be useful if the problem is nonlinear), 
% and t is the vector of Chebyshev temporal grid points on
% the current time interval.
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
%   time_error_high (def. 0.1): If the time discretization residual is
%                               higher than tol*time_error_high, increase Nt
%   time_error_low (def. 1e-4): If the time discretization residual is
%                               lower than tol*time_error_low, decrease Nt
%   max_nt (default 32):        Maximal Nt allowed. If the temporal
%                               resolution is still insufficient, split the
%                               time interval and invoke tamen on each
%                               subinterval recurrently
%   ntstep (default 8):         Increment (step) for changing Nt (also the
%                               minimal Nt).
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
% The outputs are X (on the current time interval), opts (with missed
% fields now populated with their default values), a cell array t of vectors 
% of Chebyshev time points in (0,1], and the last snapshot x=X(:,...,:,end).
% Every cell in t corresponds to a particular subinterval of the adaptive
% scheme. Use cell2mat(t) to obtain the global vector of points.
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
%       http://arxiv.org/abs/1301.6068  and
%       http://arxiv.org/abs/1304.1222

function [X,opts,t,x] = tamen(X,A,tol,opts,obs)
% Parse the solution
[d,n,~,rx,vectype]=grumble_vector(X,'x');
d = d-1; % the last block corresponds to time, distinguish it from "space"
% by treating the tensor as d+1-dimensional.
if (isa(X, 'tt_tensor'))
    X = core2cell(X);
else
    % {d,R} format
    X = X(:,end);
    rx = rx(:,end);
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
if (~isfield(opts, 'time_error_low')); opts.time_error_low=1e-4;  end;
if (~isfield(opts, 'time_error_high'));opts.time_error_high=1e-1; end;
if (~isfield(opts, 'max_nt'));         opts.max_nt=32;            end;
if (~isfield(opts, 'ntstep'));         opts.ntstep=8;             end;

% Internal use only -- these two define the time interval in a recurrent
% call
if (~isfield(opts, 'min_t'));          opts.min_t=0;              end;
if (~isfield(opts, 'max_t'));          opts.max_t=1;              end;
% Another internal -- skip the initial orthogonalization in the inner runs
if (~isfield(opts, 'reort0'));         opts.reort0=true;          end;

% Check if the number of time steps is not too small, expand the time block
% if necessary
if (n(d+1)<opts.ntstep)
    X{d+1} = [repmat(X{d+1}(:,1), 1, opts.ntstep-n(d+1)), X{d+1}];
    n(d+1) = opts.ntstep;
end;

% The local_iters parameter affects only the spatial blocks. The temporal
% system will be solved directly, so just exit after 1 dummy iteration
% inside the AMEn algorithm
if (numel(opts.local_iters)==1)
    opts.local_iters = [opts.local_iters*ones(d,1); 1];
end;

% Orthogonalize the spatial part of the solution
if (opts.reort0)
    for i=1:d-1
        crl = reshape(X{i}, rx(i)*n(i), rx(i+1));
        [crl, rv]=qr(crl, 0);
        crr = reshape(X{i+1}, rx(i+1), n(i+1)*rx(i+2));
        crr = rv*crr;
        rx(i+1) = size(crl, 2);
        X{i} = reshape(crl, rx(i), n(i), 1, rx(i+1));
        X{i+1} = reshape(crr, rx(i+1), n(i+1), 1, rx(i+2));
    end;
end;
% Save this (original) X for if we need to refine the time step
X_initial = X;
% Extract a precursor for x0, and also the spatial RHS
x0 = X;
rx0 = [rx(1:d); 1; 1];
x0{d} = reshape(x0{d}, rx(d)*n(d), rx(d+1));
% Now the last snapshot of the previous call is our x0
x0{d} = x0{d}*X{d+1}(:,end); % Now x0 is ready, and |x0|=|x0{d}|.
x0{d} = reshape(x0{d}, rx(d), n(d), 1);

% Check for not too large nt
if (n(d+1)>opts.max_nt)
    % Drop Xt. It's safe since x0 was extracted already
    n(d+1) = opts.max_nt;
    X{d+1} = ones(1, n(d+1)); % having a uniform Xt might give faster convergence
    X{d} = x0{d};
    rx(d+1)=1;
end;

% Prepare the spectral scheme in time
% Spectral differentiator
[tprev,St]=chebdiff(n(d+1)); % The time of this operation is negligible
% Temporal RHS
rhst = sum(St, 2);
rhst = reshape(rhst, 1, n(d+1));
x0{d+1} = rhst;

% Parse the matrix. We can't overwrite A, since we may call it multiple
% times below, so store the cell form as As
[As,Ra,ras]=parse_matrix(A,n,X,opts.min_t + tprev*(opts.max_t - opts.min_t),vectype);

% Temporal derivative
As{d+1,Ra+1} = St;

% x0's second norm
x0norm = norm(x0{d}, 'fro');

% Parse auxiliary enrichments
if ((nargin>=5)&&(~isempty(obs)))
    if (~isa(obs, 'cell'))
        error('Aux vectors must be given in a cell array');
    end;
    Raux = size(obs,2);
    Aux = cell(d+1,Raux);
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
        obs = Aux(1:d,:); % Might need it for recurrent calls
    else
        % Aux contains {d,R}
        [~,~,~,raux]=grumble_vector(obs,'aux',d,n(1:d));
        Aux(1:d,:) = obs;
    end;
    raux = [raux; ones(1,Raux)];
else
    Aux = [];
    raux = [];
    Raux = 0;
    obs = [];
end;

% Prepare a random initial guess for z
ZAX=[]; ZY=[];
z = cell(d+1,1);
rz = [1;opts.kickrank*ones(d,1);1];
for i=d+1:-1:2
    z{i} = randn(n(i)*rz(i+1), rz(i));
    [z{i},~]=qr(z{i}, 0);
    rz(i) = size(z{i},2);
    z{i} = z{i}.';
    z{i} = reshape(z{i}, rz(i), n(i), 1, rz(i+1));
end;
z{1} = randn(1, n(1), 1, rz(2));

% Start amen sweeps
for swp=1:opts.nswp
    % Run the spatial solver
    if (Raux>0)
        [X,rx,z,rz,ZAX,ZY,opts,errs,resids,XAX,XY,XAUX]=amenany_sweep(n, X,rx,As,ras,x0,rx0,z,rz, tol, opts, ZAX, ZY, Aux, raux);
    else
        [X,rx,z,rz,ZAX,ZY,opts,errs,resids,XAX,XY]=amenany_sweep(n, X,rx,As,ras,x0,rx0,z,rz, tol, opts, ZAX, ZY, Aux, raux);
    end;
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
        Caux = zeros(rx(d+1), Raux);
        for k=1:Raux
            Caux(:,k) = XAUX{k,d+1}; % each xaux is rxs(d+1) x 1
        end;
        [Caux,~]=qr(Caux,0);
        u0_save = Caux'*u0;
        u0_change = u0 - Caux*u0_save;
        u0_change = u0_change*sqrt(abs(x0norm^2 - norm(u0_save)^2))/norm(u0_change);
        u0 = Caux*u0_save + u0_change;
    else
        u0 = u0*(x0norm/norm(u0));
    end;
    % Spatial matrix is reduced to XAX{d+1} [rxs(d+1) x rxs(d+1) x ras(d+1)]
    At = zeros(rx(d+1)*n(d+1), rx(d+1)*n(d+1));
    % Assemble the space-time system
    for k=1:Ra
        Atk = reshape(XAX{k,d+1}, rx(d+1)*rx(d+1), ras(d+1,k));
        Atk = Atk*As{d+1,k};
        Atk = reshape(Atk, rx(d+1), rx(d+1), n(d+1), n(d+1));
        Atk = permute(Atk, [1,3,2,4]);
        Atk = reshape(Atk, rx(d+1)*n(d+1), rx(d+1)*n(d+1));
        At = At+Atk;
    end;
    % At this point, we want to estimate the residual by using a rectangular
    % Cheb matrix
    nt2 = n(d+1)+opts.ntstep;
    [t2,St2]=chebdiff(nt2); % The time of this operation is negligible
    % Construct interpolant from nt to nt2    
    P = cheb2_interpolant(tprev, t2);
    Srect = St2*P;
    % Now the matrix of size 2nt*r x nt*r
    At = kron(P, eye(rx(d+1)))*At;
    At = kron(Srect, eye(rx(d+1)))+At;
    % RHS
    rhst = sum(Srect,2);
    yt = u0*rhst.';
    yt = reshape(yt, rx(d+1)*nt2, 1);
    % Solve the LEast Squares system, obtain the temporal solution
    [Q,R]=qr(At,0);
    xt = Q'*yt;
    xt = R\xt;
    time_resid = norm(At*xt-yt)/norm(yt);
    
    if (opts.verb>0)
        fprintf('tamen: nt=%d, swp=%d, err=%3.3e, res=%3.3e, time_res=%3.3e, rank=%d\n', n(d+1), swp, max_err, max_res, time_resid, max(rx));
    end;

    X{d+1} = reshape(xt, rx(d+1), n(d+1));
    if (time_resid>tol*opts.time_error_high)
        % If the discretization is insufficient, keep the new grid
        n(d+1) = nt2;
        X{d+1} = X{d+1}*P.';        
        if (nt2>opts.max_nt)
            % Split integration to steps
            fprintf('Having more than %d Cheb polynomials is not recommended.\nHalving the time step...\n', opts.max_nt);
            % Assemble X from x0
            X = X_initial;
%             X(1:d) = x0(1:d);
%             X{d+1} = ones(1, opts.max_nt);
            if (strcmp(vectype, 'tt_tensor'))
                for i=1:d
                    r1 = size(X{i},1);
                    r2 = size(X{i},4);
                    X{i} = reshape(X{i}, r1, n(i), r2);
                end;
                X = cell2core(tt_tensor,X);
            end;
            % Prepare A*0.5 in any of possible forms
            Ahalf = A;
            if (isa(A,'tt_matrix'))
                Ahalf = 0.5*Ahalf;
            elseif (isa(A,'cell'))
                for k=1:Ra
                    Ahalf{1,k}=Ahalf{1,k}*0.5;
                end;
            else
                Ahalf = @(x,t)(A(x,t)*0.5);
            end;
            % Save my time interval for future
            min_t_my = opts.min_t;
            max_t_my = opts.max_t;
            opts.reort0 = false; % we know it comes out orthogonal
            % Call ourselves recurrently on the first half-interval
            opts.max_t = min_t_my+(max_t_my-min_t_my)*0.5;
            fprintf('Solving on (%g, %g]\n', opts.min_t, opts.max_t);
            [X1,opts,t1] = tamen(X,Ahalf,tol,opts,obs);
            % t1 is cell of all prev. time points, but not the first time
            % we are here
            if (~isa(t1, 'cell'))
                t1 = {t1};
            end;
            % Call ourselves recurrently on the second half-interval
            opts.min_t = min_t_my+(max_t_my-min_t_my)*0.5;
            opts.max_t = max_t_my;
            opts.reort0 = false; % we know it comes out orthogonal            
            % Only the last time interval is needed to continue
            if (isa(X1, 'cell'))
                X2 = X1(:,end);
            else
                % X1 is a tt_tensor
                X2 = X1; % we don't really know how many components to extract
                Xt = X1{d+1};
                X2{d+1} = Xt(:, end-numel(t1{end})+1:end);
            end;
            fprintf('Solving on (%g, %g]\n', opts.min_t, opts.max_t);
            [X2,opts,t2,x] = tamen(X2,Ahalf,tol,opts,obs);
            if (~isa(t2, 'cell'))
                t2 = {t2};
            end;
            % Restore the original values in opts
            opts.min_t = min_t_my;
            opts.max_t = max_t_my;
            if (isfield(opts, 'reort0'))
                opts = rmfield(opts, 'reort0');
            end;
            % Merge the solutions
            if (isa(X1, 'cell'))&&(isa(X2, 'cell'))
                X = [X1,X2];
            elseif (isa(X1, 'tt_tensor'))&&(isa(X2, 'tt_tensor'))
                % We need to direct sum the time TT blocks
                r1 = X1.r(d+1);
                n1 = X1.n(d+1);
                r2 = X2.r(d+1);
                n2 = X2.n(d+1);
                X1{d+1} = [X1{d+1}, zeros(r1, n2)];
                X2{d+1} = [zeros(r2, n1), X2{d+1}];
                X = X1+X2;
            else
                error('something''s wrong, can''t merge {d,R} from one interval and tt_tensor from another');
            end;
            t = [t1; t2]; % t1 and t2 are cells
            return;
        end;
    end;
    if (time_resid<tol*opts.time_error_low)&&(n(d+1)>opts.ntstep)
        % We can decrease nt
        n(d+1) = n(d+1)-opts.ntstep;
        [t2,St2]=chebdiff(n(d+1)); % The time of this operation is negligible
        P = cheb2_interpolant(tprev,t2);
        X{d+1} = X{d+1}*P.';
    end;
    % And regenerate matrix and residual if the time grid has changed    
    if (numel(tprev)~=n(d+1))
        % We really need to regenerate other A blocks. NOT interpolate!
        [As,Ra,ras]=parse_matrix(A,n,X,opts.min_t + t2*(opts.max_t - opts.min_t),vectype); 
        % New temporal RHS.
        x0{d+1} = sum(St2,2);
        % Temporal derivative
        As{d+1,Ra+1} = St2; % note: this matrix should be square!
        % Residual
        z{d+1} = reshape(z{d+1}, rz(d+1), numel(tprev));
        z{d+1} = z{d+1}*P.';
        ZAX=[]; ZY=[];
        % Time nodes
        tprev = t2;
    end;

    % Check the convergence
    if ((strcmp(opts.trunc_norm, 'fro'))&&(max_err<tol))||((~strcmp(opts.trunc_norm, 'fro'))&&(max_res<tol))
        break;
    end;
end;

% Cast the whole solution to tt_tensor if we want
if (strcmp(vectype, 'tt_tensor'))
    for i=1:d
        X{i} = reshape(X{i}, rx(i), n(i), rx(i+1));
    end;
    X = cell2core(tt_tensor, X);
end;

% Return the last snapshot
if (nargout>3)
    Xt = X{d+1};
    Xt = Xt(:,end);
    if (strcmp(vectype, 'tt_tensor'))
        x = chunk(X,1,d);
        x = x*Xt;
    else
        x = X(1:d);
        x{d} = reshape(x{d}, rx(d)*n(d), rx(d+1));
        x{d} = x{d}*Xt;
        x{d} = reshape(x{d}, rx(d), n(d));
        rx(d+1) = 1;
        for i=1:d
            x{i} = reshape(x{i}, rx(i), n(i), 1, rx(i+1)); % store the sizes in
        end;        
    end;
end;


% local_iters is normally a scalar, return it
opts.local_iters = opts.local_iters(1);

t = opts.min_t + tprev*(opts.max_t - opts.min_t); % time nodes
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
        As{i,Ra+1} = eye(n(i));
    end;
end;
% Fill temporal (d+1,:) matrix parts with identities if necessary
for k=1:Ra
    if (isempty(As{d+1,k}))
        As{d+1,k} = eye(n(d+1));
    end;
    As{d+1,k} = reshape(As{d+1,k}, ras(d+1,k), n(d+1)*n(d+1));
    As{d+1,k} = -As{d+1,k};
end;
end

