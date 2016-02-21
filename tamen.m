% Time-dependent Alternating Minimal Energy algorithm in the TT format
%   function [Xs,Xt,opts,x,t] = tamen(Xs,Xt,A,tol,opts,obs)
%
% Tries to solve the linear ODE dx/dt=Ax in the Tensor Train format using 
% the AMEn iteration for the Chebyshev discretization of the time derivative.
% The time interval is [0,1], please rescale A if necessary.
% Nevertheless, the inner splitting and discretization of this interval is
% performed adaptively. This may result in different forms of output,
% depending on how many inner splittings were necessary, see below.
%
% Xs is the spatial part of the solution, which must have the form of the
% tensor train. Input can be either a tt_tensor class from the TT-Toolbox, 
% or a cell array of size d x 1, containing TT cores (see help ttdR).
% However, output can be a d x R cell array if the time interval was split
% into R subintervals during refinement of time discretization. 
% For convenience of several subsequent invocations, the input can also have a
% d x R form with R>1, but only the latest column (:,R) will be extracted,
% since it is the column where the last snapshot from the previous run is
% located.
% If Xs was given as tt_tensor and many time intervals were necessary, the
% output will still be a single tt_tensor, obtained by calling horzcat on
% tt_tensors from different subintervals. !!NOTE!! This can lead to
% inefficient storage, since no truncation is applied. Consider using the 
% {d,R} format instead if you expect your matrix to be stiff.
% 
% Xt is the temporal part of the solution, being a r x Nt double matrix,
% where r is the sum of the last TT ranks of Xs(d,:), and Nt is the time 
% grid size. On input, the initial state of the ODE is extracted as the last
% snapshot of the given {Xs,Xt} solution, i.e. x0 = Xs(:,end)*Xt(:,end). 
% For convenience in the first run, it is possible to pass x0 in Xs, and 
% in this case Xt must be a 1x1 cell containing the number of Chebyshev 
% nodes in time, i.e. Xt={Nt}. The output is always a r x Nt matrix, so you
% may simple re-invoke the method for the next time steps.
% The vector of time points is returned in the output t
% in the acscending order, such that t(end)==1.
% If many time subintervals were necessary, t contains correct global time
% points on (0,1], merged from all intervals. The columns in Xt are
% consistent with t. However, note that the rows of Xt are only consistent
% with the TOTAL rank of Xs, not necessarily with the ranks of individual
% components Xs(:,k).
%
% Alternatively, one can avoid a separate Xt and keep the temporal dimension
% as the last rank in Xs. In this case, Xt should be empty [] on input, and
% it is also returned empty on output. The last rank of Xs corresponds to
% the time dimension on both input and output.
% If many time subintervals were necessary, Xt is also empty, since each of
% Xs columns carries its own time points.
%
% A must be a square nonpositive definite matrix in the TT format, either a
% tt_matrix class from the TT-Toolbox, or a cell array of size D x R,
% containing TT cores (see help ttdR), or a function, constructing A in
% either of the previous two formats. A can be either stationary, in this
% case D==d, and A contains the spatial part only, or time-dependent, in
% this case D==d+1, the first d blocks contain the spatial part, and the
% last block must be of size r x Nt x Nt, diagonal w.r.t. the Nt x Nt part.
% The whole matrix is thus diagonal w.r.t. the time, and the diagonal 
% contains the spatial matrices evaluated at the Chebyshev points, A(t_j)
%
% If A is a function, it should take three arguments Xs,Xt,t. The first two
% represent the solution as described above (might be useful if the problem
% is nonlinear), and t is the vector of Chebyshev temporal grid points on
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
%   ntstep (default 8):         A step for changing the number of time grid
%                               points (Nt) during adaptation
%   time_error_low (def. 1e-4): If the time discretization residual drops
%                               below tol*time_error_low, decrease Nt
%   max_nt (default 64):        Maximal Nt allowed. If the temporal
%                               resolution is still insufficient, split the
%                               time interval and invoke tamen on each
%                               subinterval recurrently
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
% Also, the last snapshot x=Xs*Xt(:,end) is returned for convenience,
% especially in the {d,R} format.
% The last output t is the vector of Chebyshev time points in (0,1].
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

function [Xs,Xt,opts,x,t] = tamen(Xs,Xt,A,tol,opts,obs)
% Parse the solution
% Temporal part
if (isa(Xt, 'cell'))
    % Allow users to specify Xt={Nt} in the first run
    nt = Xt{1};
    Xt = [zeros(1,nt-1), 1];
elseif (isa(Xt, 'double'))
    nt = size(Xt,2);
end;
% Spatial part
[d,n,~,rxs,vectype]=grumble_vector(Xs,'x');
% Check if Xt is incorporated into xs    
Xt_in_Xs = false;
if (isempty(Xt))    
    Xt_in_Xs = true;
    nt = rxs(d+1);
    Xt = eye(nt);
end;
% Copy xs to expanded storage
rxs = [rxs; 1];
xs = cell(d+1,1);
xs{d+1} = Xt;
if (isa(Xs, 'tt_tensor'))
    xs(1:d) = core2cell(Xs);
else
    % {d,R} format
    xs(1:d) = Xs(:,end);
end;
% The total size of the system
N = [n; nt]; 

% Parse the options
if (nargin<5)||(isempty(opts))
    opts = struct;
end;
% Parse opts parameters. We just populate what we do not have by defaults
if (~isfield(opts, 'nswp'));           opts.nswp=20;              end;
if (~isfield(opts, 'kickrank'));       opts.kickrank=4;           end;
if (~isfield(opts, 'verb'));           opts.verb=1;               end;
if (~isfield(opts, 'local_iters'));    opts.local_iters=100;      end;
if (~isfield(opts, 'trunc_norm'));     opts.trunc_norm='fro';     end;
if (~isfield(opts, 'time_error_low')); opts.time_error_low=1e-4;  end;
if (~isfield(opts, 'max_nt'));         opts.max_nt=64;            end;
if (~isfield(opts, 'ntstep'));         opts.ntstep=8;             end;

% Internal use only -- these two define the time interval in a recurrent
% call
if (~isfield(opts, 'min_t'));          opts.min_t=0;              end;
if (~isfield(opts, 'max_t'));          opts.max_t=1;              end;

% Check if the number of time steps is not too small
if (nt<opts.ntstep)
    Xt = [zeros(size(Xt,1), opts.ntstep-nt), Xt];
    nt = opts.ntstep;
    N(d+1)=nt;
    xs{d+1} = Xt;
end;

% The local_iters parameter affects only the spatial blocks. The temporal
% system will be solved directly, so just exit after 1 dummy iteration
% inside the AMEn algorithm
if (numel(opts.local_iters)==1)
    opts.local_iters = [opts.local_iters*ones(d,1); 1];
end;

% Orthogonalize the spatial part of the solution
for i=1:d-1
    crx = reshape(xs{i}, rxs(i)*n(i), rxs(i+1));
    [crx, rvs]=qr(crx, 0);
    cr2 = reshape(xs{i+1}, rxs(i+1), n(i+1)*rxs(i+2));
    cr2 = rvs*cr2;
    rxs(i+1) = size(crx, 2);
    xs{i} = reshape(crx, rxs(i), n(i), 1, rxs(i+1));
    xs{i+1} = reshape(cr2, rxs(i+1), n(i+1), 1, rxs(i+2));
end;
% Extract a precursor for x0, and also the spatial RHS
x0 = xs;
rx0 = [rxs(1:d); 1; 1];
x0{d} = reshape(x0{d}, rxs(d)*n(d), rxs(d+1));
% Only take the _last_ rxs(d+1) components of Xt, since the latter can come
% from merging of several time intervals
if (size(Xt,1)<rxs(d+1))
    error('Mismatching Xt and Xs. Are you passing Xt={Nt} to the second call?');
end;
x0{d} = x0{d}*Xt(end-rxs(d+1)+1:end,end); % Now x0 is ready, and |x0|=|x0{d}|.
x0{d} = reshape(x0{d}, rxs(d), n(d), 1);

% Check for not too large nt
if (nt>opts.max_nt)
    % Drop Xt. It's safe since x0 was extracted already
    nt = opts.max_nt;
    N(d+1)=nt;
    Xt = [zeros(1,nt-1), 1];
    xs{d} = x0{d};
    rxs(d+1)=1;
    xs{d+1} = Xt;
end;

% Prepare the spectral scheme in time
% Spectral differentiator
[tprev,St]=chebdiff(nt); % The time of this operation is negligible
% Temporal RHS.
rhst = sum(St, 2);
rhst = reshape(rhst, 1, nt);
x0{d+1} = rhst;

% Parse the matrix
[As,Ra,ras]=parse_matrix(A,N,xs,opts.min_t + tprev*(opts.max_t - opts.min_t),vectype);

% Temporal derivative
As{d+1,Ra+1} = St;

% Measure the second norm
x0norm = norm(x0{d}, 'fro');

% Parse auxiliary enrichments
if ((nargin>=6)&&(~isempty(obs)))
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
                [~,~,~,raux(:,i)]=grumble_vector(obs{i},'aux',d,n);
                Aux(1:d,i) = core2cell(obs{i});
            else
                error('All aux vectors must be either tt_tensors or {d,R}s');
            end;
        end;
        obs = Aux(1:d,:); % Might need it for recurrent calls
    else
        % Aux contains {d,R}
        [~,~,~,raux]=grumble_vector(obs,'aux',d,n);
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
ZAXs=[]; ZYs=[];
zs = cell(d+1,1);
rzs = [1;opts.kickrank*ones(d,1);1];
for i=d+1:-1:2
    zs{i} = randn(N(i)*rzs(i+1), rzs(i));
    [zs{i},~]=qr(zs{i}, 0);
    rzs(i) = size(zs{i},2);
    zs{i} = zs{i}.';
end;
zs{1} = randn(N(1), rzs(2));

% Start amen sweeps
for swp=1:opts.nswp
    % Run the spatial solver
    if (Raux>0)
        [xs,rxs,zs,rzs,ZAXs,ZYs,opts,errs,resids,XAX,XY,XAUX]=amenany_sweep(N, xs,rxs,As,ras,x0,rx0,zs,rzs, tol, opts, ZAXs, ZYs, Aux, raux);
    else
        [xs,rxs,zs,rzs,ZAXs,ZYs,opts,errs,resids,XAX,XY]=amenany_sweep(N, xs,rxs,As,ras,x0,rx0,zs,rzs, tol, opts, ZAXs, ZYs, Aux, raux);
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
        Caux = zeros(rxs(d+1), Raux);
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
    At = zeros(rxs(d+1)*nt, rxs(d+1)*nt);
    % Assemble the space-time system
    for k=1:Ra
        Atk = reshape(XAX{k,d+1}, rxs(d+1)*rxs(d+1), ras(d+1,k));
        Atk = Atk*As{d+1,k};
        Atk = reshape(Atk, rxs(d+1), rxs(d+1), nt, nt);
        Atk = permute(Atk, [1,3,2,4]);
        Atk = reshape(Atk, rxs(d+1)*nt, rxs(d+1)*nt);
        At = At+Atk;
    end;
    % At this point, we want to measure the residual by using a rectangular
    % Cheb matrix
    nt2 = nt+opts.ntstep;
    [t2,St2]=chebdiff(nt2); % The time of this operation is negligible
    % Construct interpolant from nt to nt2    
    P = lagrange_interpolant(tprev,t2);
    Srect = St2*P;
    % Now the matrix of size 2nt*r x nt*r
    At = kron(P, eye(rxs(d+1)))*At;
    At = kron(Srect, eye(rxs(d+1)))+At;
    % RHS
    rhst = sum(Srect,2);
    yt = u0*rhst.';
    yt = reshape(yt, rxs(d+1)*nt2, 1);
    % Solve the LEast Squares system, obtain the temporal solution
    [Q,R]=qr(At,0);
    xt = Q'*yt;
    xt = R\xt;
    time_resid = norm(At*xt-yt)/norm(yt);
    
    if (opts.verb>0)
        fprintf('tamen: nt=%d, swp=%d, err=%3.3e, res=%3.3e, time_res=%3.3e, rank=%d\n', nt, swp, max_err, max_res, time_resid, max(rxs));
    end;

    Xt = reshape(xt, rxs(d+1), nt);
    if (time_resid>tol)
        % If the discretization is insufficient, keep the new grid
        nt = nt2;
        Xt = Xt*P.';        
        if (nt2>opts.max_nt)
            % Split integration to steps
            fprintf('Having more than %d Cheb polynomials is not recommended.\nHalving the time step...\n', opts.max_nt);
            % Assemble x0
            Xs = x0(1:d);
            if (strcmp(vectype, 'tt_tensor'))
                for i=1:d
                    r1 = size(Xs{i},1);
                    r2 = size(Xs{i},4);
                    Xs{i} = reshape(Xs{i}, r1, n(i), r2);
                end;
                Xs = cell2core(tt_tensor,Xs);
            end;
            % Check if Xt is carried in Xs
            if (Xt_in_Xs)
                Xt = [zeros(1,opts.max_nt-1), 1];
                if (strcmp(vectype, 'tt_tensor'))
                    Xs = Xs*Xt;
                else
                    Xs{d} = reshape(Xs{d}, [], 1);
                    Xs{d} = Xs{d}*Xt;
                    Xs{d} = reshape(Xs{d}, [], n(d), 1, opts.max_nt);
                end;
                Xt = [];
            else
                Xt = {opts.max_nt};
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
                Ahalf = @(xs,xt,t)(A(xs,xt,t)*0.5);
            end;
            % Save my time interval for future
            min_t_my = opts.min_t;
            max_t_my = opts.max_t;
            % Call ourselves recurrently on the first half-interval
            opts.max_t = min_t_my+(max_t_my-min_t_my)*0.5;
            fprintf('Solving on (%g, %g]\n', opts.min_t, opts.max_t);
            [Xs1,Xt1,opts,~,t1] = tamen(Xs,Xt,Ahalf,tol,opts,obs);
            % Call ourselves recurrently on the second half-interval
            opts.min_t = min_t_my+(max_t_my-min_t_my)*0.5;
            opts.max_t = max_t_my;
            fprintf('Solving on (%g, %g]\n', opts.min_t, opts.max_t);
            [Xs2,Xt2,opts,x,t2] = tamen(Xs1,Xt1,Ahalf,tol,opts,obs);
            % Restore the values in opts
            opts.min_t = min_t_my;
            opts.max_t = max_t_my;
            % Merge the solutions
            Xs = [Xs1,Xs2];
            if (~isempty(Xt))
                Xt = [Xt1, zeros(size(Xt1,1), numel(t2));
                    zeros(size(Xt2,1), numel(t1)), Xt2];
            end;
            t = [t1;t2];
            return;
        end;
    end;
    if (time_resid<tol*opts.time_error_low)&&(nt>opts.ntstep)
        % We can decrease nt
        nt = nt-opts.ntstep;
        [t2,St2]=chebdiff(nt); % The time of this operation is negligible
        P = lagrange_interpolant(tprev,t2);
        Xt = Xt*P.';
    end;
    % Store the solution
    xs{d+1} = Xt;
    if (numel(tprev)~=nt)
        % And regenerate other items if the time grid changed
        N(d+1) = nt;
        % We also need to regenerate other A blocks. NOT interpolate!
        [As,Ra,ras]=parse_matrix(A,N,xs,opts.min_t + t2*(opts.max_t - opts.min_t),vectype); 
        % Temporal RHS.
        x0{d+1} = sum(St2,2);
        % Temporal derivative
        As{d+1,Ra+1} = St2; % note: this matrix should be square!
        % Residual
        zs{d+1} = reshape(zs{d+1}, rzs(d+1), numel(tprev));
        zs{d+1} = zs{d+1}*P.';
        ZAXs=[]; ZYs=[];
        % Time nodes
        tprev = t2;
    end;
        
    if (strcmp(opts.trunc_norm, 'fro'))
        if (max_err<tol); break; end;
    else
        if (max_res<tol); break; end;
    end;
end;

% Cast spatial solution to the desired form
if (strcmp(vectype, 'tt_tensor'))
    for i=1:d
        xs{i} = reshape(xs{i}, rxs(i), n(i), rxs(i+1));
    end;
    Xs = cell2core(tt_tensor, xs(1:d));
else
    Xs = xs(1:d);
%     for i=1:d
%         Xs{i} = reshape(Xs{i}, rxs(i), n(i), 1, rxs(i+1)); % store the sizes in
%     end;
end;


% Return the last snapshot
if (nargout>3)
    x = xs(1:d);
    x{d} = reshape(x{d}, rxs(d)*n(d), rxs(d+1));
    x{d} = x{d}*Xt(:,end);
    x{d} = reshape(x{d}, rxs(d), n(d));
    % Cast the solution to the desired form
    if (strcmp(vectype, 'tt_tensor'))
        x = cell2core(tt_tensor, x);
%     else
%         rxsd = rxs(d+1);
%         rxs(d+1) = 1;
%         for i=1:d
%             x{i} = reshape(x{i}, rxs(i), n(i), 1, rxs(i+1)); % store the sizes in
%         end;
%         rxs(d+1) = rxsd;
    end;
end;

% Do we want to have Xt separately?
if (Xt_in_Xs)
    if (strcmp(vectype, 'tt_tensor'))
        Xs = Xs*Xt;
    else
        Xs{d} = reshape(Xs{d}, rxs(d)*n(d), rxs(d+1));
        Xs{d} = Xs{d}*Xt;
        Xs{d} = reshape(Xs{d}, rxs(d), n(d), 1, nt); % store the sizes in
    end;
    Xt = [];
end;


% local_iters is normally a scalar, return it
opts.local_iters = opts.local_iters(1);

t = opts.min_t + tprev*(opts.max_t - opts.min_t); % time nodes
end

%Computes the Spectral Chebyshev differentiation matrix and nodes
%on the interval [0,1]. Used Dirichlet BC at 0, and, t is sorted acsending
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


function [As,Ra,ras]=parse_matrix(A,N,xs,t,vectype)
% Parse the matrix. It will be harder than in amen_solve, since we may have
% either stationary or time-dependent matrix, or a function generating
% either of those

% Extract the sizes
d = numel(N)-1;
n = N(1:d);
nt = N(d+1);

if ((~isa(A, 'cell'))&&(~isa(A, 'tt_matrix')))
    % This is a function and takes (Xs,Xt,t)
    % Cast spatial solution to the desired form    
    Xs = xs(1:d);
    if (strcmp(vectype, 'tt_tensor'))
        for i=1:d
            r1 = size(Xs{i},1);
            r2 = size(Xs{i},4);
            Xs{i} = reshape(Xs{i}, r1, n(i), r2);
        end;        
        Xs = cell2core(tt_tensor, Xs);
    end;
    A = A(Xs,xs{d+1},t);
end;

% Then, get the dimension of the matrix
if (isa(A, 'tt_matrix'))
    D = A.d;
else
    D = size(A,1);
end;
% The only allowed dimensions are d and d+1, where d is the dimension of
% the vector
switch(D)
    case d
        % Matrix is stationary
        [~,~,Ra,ras]=grumble_matrix(A,'A',d,n);
        ras = [ras, ones(d+1,1)];
        ras = [ras; ones(1,Ra+1)];
    case d+1
        % Matrix is time-dependent
        [~,~,Ra,ras]=grumble_matrix(A,'A',d+1,N);
        ras = [ras, ones(d+2,1)];
    otherwise
        error('dim(A) must be either d (stationary matrix) or d+1 (time-dependent)');
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
        As{d+1,k} = eye(nt);
    end;
    As{d+1,k} = reshape(As{d+1,k}, ras(d+1,k), nt*nt);
    As{d+1,k} = -As{d+1,k};
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
        fprintf('Tensor Chain format (R>1) is not allowed for input %s,\nbut it can be due to multiple time intervals.\nExtracting the last column.\n', xname);
        x = x(:,end);
        Rx = 1;
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
