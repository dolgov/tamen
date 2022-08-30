% One AMEn iteration for the linear system solution.
%   function [x,rx,z,ZAX,ZY,opts,errs,resids,XAX,XY,XAUX]=amenany_sweep(x,A,y,z, tol, opts, ZAX, ZY, aux)
%
% !!! Note !!!!
% This is a technical routine, please use higher-level amen_solve or tamen 
% unless you understand precisely what are you doing.

function [x,rx,z,ZAX,ZY,opts,errs,resids,XAX,XY,XAUX]=amenany_sweep(x,A,y,z, tol, opts, ZAX, ZY, aux)
% XAX: cell(Ra, d+1), XY: cell(Ry, d+1), the projections of A and Y onto X
% ZAX: cell(Ra, d+1), ZY: cell(Ry, d+1), analogously for the residual
% XAUX: cell(Raux, d+1) stores the projection of AUX.

% Parse the optional inputs
if (nargin<6)||(isempty(opts))    
    opts = struct;
end;
if (nargin<9)    
    aux = [];
end;
if (nargin<7)    
    ZAX=[];
end;
if (nargin<8)
    ZY=[];
end;

% Parse the inputs & check for consistency
[d,n,~,rx]=grumble_vector(x,'x');
[~,~,Ra,ra]=grumble_matrix(A,'a',d,n);
[~,~,Ry]=grumble_vector(y,'y',d,n);
[~,~,Rz,rz]=grumble_vector(z,'z',d,n);
[~,~,Raux,raux]=grumble_vector(aux,'o');

% Adjust the tolerance to the sqrt(d) factor of the recurrent TT-SVD
loctol = tol/sqrt(d);

% Parse opts parameters. We just populate what we do not have by defaults
if (~isfield(opts, 'resid_damp'));      opts.resid_damp=4;      end;
if (~isfield(opts, 'rmax'));            opts.rmax=Inf;          end;
if (~isfield(opts, 'max_full_size'));   opts.max_full_size=50;  end;
if (~isfield(opts, 'local_iters'));     opts.local_iters=100;   end;
if (~isfield(opts, 'trunc_norm'));      opts.trunc_norm='fro';  end;
if (~isfield(opts, 'kickrank2'));       opts.kickrank2=0;       end;
if (~isfield(opts, 'verb'));            opts.verb=1;            end;
if (~isfield(opts, 'local_prec'));      opts.local_prec='';     end;

% More parsing: check what inputs we have, and what reductions we need
if (Ra>0)
    XAX = cell(1,d+1);
    XAX{1} = 1;
    XAX{d+1} = 1;
    XAX = repmat(XAX, Ra, 1);
end;
if (Ry>0)
    XY = repmat(XAX(1,:), Ry, 1);
end;
XAUX = [];
if (Raux>0)
    XAUX = repmat(XAX(1,:), Raux, 1);
end;

% Create ZAX, ZY if they were not passed
if (isempty(ZAX))
    ZAX = repmat(XAX(1,:), Ra, 1);
end;
if (isempty(ZY))
    ZY = repmat(XAX(1,:), Ry, 1);
end;

% Right to left iteration: warmup
for i=d:-1:2
    if (Rz>0)
        if (isempty(ZY{1,i}))
            % If we have no projection, just use the previous Z block
            crzy = reshape(z{i}, rz(i)*n(i)*rz(i+1), 1);
        else
            % ZY was passed.
            % We assume that X is left-orthogonal, so we may compute the update for Z
            % Project the residual to the Z-block:
            %   project y
            crzy = assemble_local_vector(ZY(:,i), y(i,:), ZY(:,i+1));
        end
        if (isempty(ZAX{1,i}))
            % No projection - No update
            crzAx = zeros(rz(i)*n(i)*rz(i+1), 1);
        else
            % ZAX was passed.
            %   project Ax
            Ai = A(i,:);
            for k=1:Ra
                Ai{k} = reshape(Ai{k}, ra(i,k)*n(i), n(i)*ra(i+1,k));
            end;
            crzAx = local_matvec(x{i}, rx(i),n(i),rx(i+1),1, rz(i),n(i),rz(i+1), ZAX(:,i), Ai, ZAX(:,i+1), Ra,ra(i,:),ra(i+1,:));
        end
        % Residual is here
        crz = crzy-crzAx;
        crz = reshape(crz, rz(i), n(i)*rz(i+1));
        if (opts.kickrank2>0)
            % Apply the secondary enrichment (for the residual itself)
            % using random vectors.
            [~,~,crz]=svd(crz, 'econ');
            crz = crz(:,1:max(rz(i)-opts.kickrank2, 1))';
            crz2 = [crz; randn(opts.kickrank2, n(i)*rz(i+1))];
            crz = crz2;
        end;
        % Orthogonalize and store
        crz = reshape(crz, rz(i), n(i), 1, rz(i+1));
        [z{i-1},z{i},rz(i)] = orthogonalise_block(z{i-1},crz,-1);
    end;
    
    % Orthogonalization for X
    [x{i-1},x{i},rx(i)] = orthogonalise_block(x{i-1},x{i},-1);
    
    % Compute reductions
    % With X
    XAX(:,i) = rightreduce_matrix(XAX(:,i+1), x{i}, A(i,:), x{i});
    XY(:,i) = rightreduce_vector(XY(:,i+1), x{i}, y(i,:));
    % With Z
    if (Rz>0)
        ZAX(:,i) = rightreduce_matrix(ZAX(:,i+1), z{i}, A(i,:), x{i});
        ZY(:,i) = rightreduce_vector(ZY(:,i+1), z{i}, y(i,:));
    end;
end;

% Initialize the error and residual storages
errs = zeros(d,1);
resids = zeros(d,1);

% Left to right AMEn iteration: solution
for i=1:d
    % Prepare the local system
    sol_prev = reshape(x{i}, rx(i)*n(i)*rx(i+1),1);
    lociters = opts.local_iters;
    % Extract the current number of iterations wisely
    if (numel(lociters)>1)
        lociters = lociters(i);
    end;
    % Right hand side
    rhs = assemble_local_vector(XY(:,i), y(i,:), XY(:,i+1));
    norm_rhs = norm(rhs);
    % Extract the matrix parts, accelerate a plenty of iterations with them
    XAX1 = XAX(:,i);
    Ai = A(i,:);
    XAX2 = XAX(:,i+1);
    ra1 = ra(i,:);
    ra2 = ra(i+1,:);
    for k=1:Ra
        Ai{k} = reshape(Ai{k}, ra1(k)*n(i), n(i)*ra2(k));
    end;
    % Matvec function. We must pass all sizes, since extracting them in
    % every iteration is too expensive
    mvfun = @(x)local_matvec(x, rx(i),n(i),rx(i+1),1, rx(i),n(i),rx(i+1), XAX1, Ai, XAX2, Ra,ra1,ra2);
    % Measure the residual
    resids(i) = norm(rhs-mvfun(sol_prev))/norm_rhs;
    
    if (lociters>0)        
        if (rx(i)*n(i)*rx(i+1)<opts.max_full_size)
            % If the system size is small, assemble the full system and solve
            % directly
            [B,sparseflag] = assemble_local_matrix(XAX1, A(i,:), XAX2);
            if (sparseflag)
                % Permute the indices such that the spatial mode is the senior,
                % since usually it is large and sparsified, but the rank modes
                % are dense.
                rhs = reshape(rhs, rx(i)*n(i), rx(i+1));
                rhs = rhs.';
                rhs = reshape(rhs, rx(i+1)*rx(i)*n(i), 1);
            end;
            sol = B\rhs;
            if (sparseflag)
                % Recover the dimension order back
                sol = reshape(sol, rx(i+1), rx(i)*n(i));
                sol = sol.';
                sol = reshape(sol, rx(i)*n(i)*rx(i+1), 1);
                rhs = reshape(rhs, rx(i+1), rx(i)*n(i));
                rhs = rhs.';
                rhs = reshape(rhs, rx(i)*n(i)*rx(i+1), 1);
            end;
            % Report if necessary
            if (opts.verb>1)
                fprintf('\tamen_sweep: i=%d. Mldivide: sparse=%d. ', i, sparseflag);
            end;
        else
            % System is large, solve iteratively
            % Run the bicgstab without a preconditioner first
            [sol,flg,relres,iter] = bicgstab(mvfun, rhs, max(loctol/opts.resid_damp,eps*1e3), lociters, [], [], sol_prev);
            % Report
            if ((opts.verb>1)&&(flg==0))
                fprintf('\tamen_sweep: i=%d. Bicgstab: %g iters, residual %3.3e. ', i, iter, relres);
            end;
            % If we did not converge...
            if (flg>0)
                precfun = [];
                % ... use the Block Jacobi preconditioner, if required
                if (strcmp(opts.local_prec, 'r'))
                    P = assemble_local_rjacobi(XAX1, A(i,:), XAX2);
                    precfun = @(x)local_precvec(x, rx(i),n(i),rx(i+1),1, P);
                end;
                % In any case, run the bicg once again
                [sol,~,relres,iter] = bicgstab(mvfun, rhs, max(loctol/opts.resid_damp,eps*1e3), lociters, precfun, [], sol);
                % And report its performance
                if (opts.verb>1)
                    fprintf('\tamen_sweep: i=%d. Bicgstab(prec): %g iters, residual %3.3e. ', i, lociters+iter, relres);
                end;
            end;
        end;
    else
        sol = sol_prev;
    end;
    
    % Measure the error
    errs(i) = norm(sol-sol_prev)/norm(sol);
    
    % Truncation
    sol = reshape(sol, rx(i)*n(i), rx(i+1));
    if (loctol>0)&&(i<d)&&((Rz+Raux)>0)
        [u,s,v]=svd(sol,'econ');
        s = diag(s);
        if (strcmp(opts.trunc_norm, 'fro')||(lociters==0))
            % We are happy with the Frobenius-norm truncation
            sum_s=cumsum(s(end:-1:1).^2);
            r = find(sum_s>=(loctol*norm(s)).^2, 1);
            if (isempty(r))
                r = numel(s);
            else
                r = numel(s)-r+1;
            end;
        else
            % Residual truncation strategy
            % Compute res_new -- our limit for truncation
            res_new = norm(rhs-mvfun(sol))/norm_rhs;
            % Start from the previous rank...
            r = min(rx(i)*n(i),rx(i+1));
            r = max(r-rz(i+1,1), 1);
            cursol = u(:,1:r)*diag(s(1:r))*v(:,1:r)';
            % ... check the corresponding residual
            res = norm(rhs-mvfun(cursol))/norm_rhs;
            if (res<max(loctol, res_new*opts.resid_damp))
                drank = -1; % rank is overestimated, decrease it
            else
                drank = 1; % residual is large; increase the rank
            end;
            while (r>0)&&(r<=numel(s))
                % Pick the ranks in the selected direction one by one
                cursol = u(:,1:r)*diag(s(1:r))*v(:,1:r)';
                res = norm(rhs-mvfun(cursol))/norm_rhs;
                if (drank>0)
                    if (res<max(loctol, res_new*opts.resid_damp))
                        break;
                    end;
                else
                    if (res>=max(loctol, res_new*opts.resid_damp))
                        break;
                    end;
                end;
                r = r+drank;
            end;
            if (drank<0)
                % We stopped when the residual became larger than tolerance,
                % take the next value
                r=r+1;
            end;
        end;
        % Limit to the max. rank if necessary
        r = min(r, numel(s));
        r = min(r, opts.rmax);
        u = u(:,1:r);
        v = diag(s(1:r))*v(:,1:r)';
        sol = u*v; % this truncated solution will be used to compute enrichies
    else
        % No truncation requested, just orthogonalize
        [u,v]=qr(sol,0);
        r = size(u,2);
    end;
    
    % Report the chosen rank
    if (opts.verb>1)       
        fprintf('Rank: %d\n', r);
    end;
    
    % Update the residual Z
    if (Rz>0)
        % y
        crzy = assemble_local_vector(ZY(:,i), y(i,:), ZY(:,i+1));
        % Ax
        crzAx = local_matvec(sol, rx(i),n(i),rx(i+1),1, rz(i),n(i),rz(i+1), ZAX(:,i), Ai, ZAX(:,i+1), Ra,ra1,ra2);
        % z=y-Ax
        crz = crzy-crzAx;
        crz = reshape(crz, rz(i)*n(i), rz(i+1));
        if (opts.kickrank2>0)&&(i<d)
            % Secondary random enrichment, if required
            [crz,~,~]=svd(crz, 'econ');
            crz = crz(:,1:max(rz(i+1,1)-opts.kickrank2, 1));
            crz2 = [crz, randn(rz(i,1)*n(i), opts.kickrank2)];
            crz = crz2;
        end;       
        % Careful: store the old rank of z, since it is that will be used
        % in the solution enrichment, not the updated value after the QR
        rzold = rz(i+1);
        crz = reshape(crz, rz(i), n(i), 1, rz(i+1));
        [~,z{i},rz(i+1)] = orthogonalise_block([],crz,1);
    end;
    
    if (i<d)
        % Apply enrichment for the solution
        if (Rz>0)
            % y
            crzy = assemble_local_vector(XY(:,i), y(i,:), ZY(:,i+1));
            % Ax
            crzAx = local_matvec(sol, rx(i),n(i),rx(i+1),1, rx(i),n(i),rzold, XAX1, Ai, ZAX(:,i+1), Ra,ra1,ra2);
            % z
            crz = crzy-crzAx;
            crz = reshape(crz, rx(i)*n(i), rzold);            
            u2 = [u,crz]; % Enrichment is made here
            u = u2;
        end;
        % Add auxiliary vectors if there are
        for k=1:Raux
            craux = reshape(aux{i,k}, raux(i,k), n(i)*raux(i+1,k));
            craux = XAUX{k,i}*craux;
            craux = reshape(craux, rx(i)*n(i), raux(i+1,k));
            u2 = [u, craux];
            u = u2;
        end;
        if (size(u,2)>r)
            % Several enrichments have been made, QR is needed
            [u,rv]=qr(u, 0);
            % Extract XAUX from the rv factor
            rvx = rv(:,1:r); % This is for the solution
            if (Rz>0)
                % If Z is there, remove it also
                rv = rv(:,(r+rzold+1):size(rv,2));
            else
                rv = rv(:,(r+1):size(rv,2));
            end;
            % Now rv contains only the aux-parts. Extract 'em
            for k=1:Raux
                XAUX{k,i+1} = rv(:,1:raux(i+1,k));                     %#ok
                rv = rv(:, (raux(i+1,k)+1):size(rv,2));
            end;
            % Restore the solution factor. We love White's prediction
            rv = rvx;            
            v = rv*v;
            r = size(u,2);
        end;
        % Store u and calculate the White's prediction for the next block
        cr2 = x{i+1};
        cr2 = reshape(cr2, rx(i+1), n(i+1)*rx(i+2));
        cr2 = v*cr2;
        rx(i+1) = r;
        x{i+1} = reshape(cr2, rx(i+1), n(i+1), 1, rx(i+2)); % now it is a good initial guess
        x{i} = reshape(u, rx(i), n(i), 1, rx(i+1));
        
        % Update reductions
        % For X
        XAX(:,i+1) = leftreduce_matrix(XAX(:,i), x{i}, Ai, x{i});
        XY(:,i+1) = leftreduce_vector(XY(:,i), x{i}, y(i,:));
        % For Z
        if (Rz>0)
            ZAX(:,i+1) = leftreduce_matrix(ZAX(:,i), z{i}, Ai, x{i});
            ZY(:,i+1) = leftreduce_vector(ZY(:,i), z{i}, y(i,:));
        end;
    else
        x{i} = reshape(sol, rx(i), n(i), 1, rx(i+1));
    end;
end;

end



% Accumulates the left reduction W{1:k}'*A{1:k}*X{1:k}
function [WAX2] = leftreduce_matrix(WAX1, w, A, x)
% Left WAX has the form of the first matrix TT block, i.e. [rw, rx, ra]
WAX2 = WAX1;
[rx1,m,~,rx2] = size(x);
[rw1,n,~,rw2] = size(w);
wc = reshape(w, rw1, n*rw2);
xc = reshape(x, rx1*m, rx2);
Ra = size(A,2);
for k=1:Ra
    ra1 = size(WAX1{k},3);
    WAX2{k} = reshape(WAX2{k}, rw1, rx1*ra1);
    WAX2{k} = wc'*WAX2{k}; % size n rw2 x rx1 ra1
    WAX2{k} = reshape(WAX2{k}, n, rw2*rx1*ra1);
    WAX2{k} = WAX2{k}.';
    WAX2{k} = reshape(WAX2{k}, rw2*rx1, ra1*n);
    tmp = reshape(A{k}, ra1*n, []); % m*ra2(k)
    WAX2{k} = WAX2{k}*tmp; % size rw2 rx1 m ra2
    WAX2{k} = reshape(WAX2{k}, rw2, []); % rx1*m*ra2(k)
    WAX2{k} = WAX2{k}.';
    WAX2{k} = reshape(WAX2{k}, rx1*m, []); % ra2(k)*rw2
    WAX2{k} = xc.'*WAX2{k}; % size rx2, ra2 rw2
    WAX2{k} = reshape(WAX2{k}, [], rw2); % rx2*ra2(k)
    WAX2{k} = WAX2{k}.';
    WAX2{k} = reshape(WAX2{k}, rw2, rx2, []); % ra2(k)
end;
end

% Accumulates the left reduction W{1:k}'*X{1:k}
function [WX2] = leftreduce_vector(WX1, w, x)
% Left WX has the form of the first vector TT block, i.e. [rw, rx]
WX2 = WX1;
[rw1,n,~,rw2] = size(w);
wc = reshape(w, rw1, n*rw2);
Rx = size(x,2);
for k=1:Rx
    [rx1,n,rx2] = size(x{k});
    WX2{k} = wc'*WX2{k}; % size n rw2 x rx1
    WX2{k} = reshape(WX2{k}, n, rw2*rx1);
    WX2{k} = WX2{k}.';
    WX2{k} = reshape(WX2{k}, rw2, rx1*n);
    tmp = reshape(x{k}, rx1*n, rx2);
    WX2{k} = WX2{k}*tmp; % size rw2, rx2
end;
end

% Accumulates the right reduction W{k:d}'*A{k:d}*X{k:d}
function [WAX1] = rightreduce_matrix(WAX2, w, A, x)
% Right WAX has the form of the last matrix TT block, i.e. [ra, rw, rx]
WAX1 = WAX2;
[rx1,m,~,rx2] = size(x);
[rw1,n,~,rw2] = size(w);
wc = conj(w);
wc = reshape(wc, rw1, n*rw2);
xc = reshape(x, rx1*m, rx2);
Ra = size(A,2);
for k=1:Ra
    ra2 = size(WAX2{k},1);
    WAX1{k} = reshape(WAX1{k}, ra2*rw2, rx2);
    WAX1{k} = xc*WAX1{k}.'; % size rx1 m x ra2 rw2
    WAX1{k} = reshape(WAX1{k}, [], m*ra2*rw2); % rx1
    WAX1{k} = WAX1{k}.';
    WAX1{k} = reshape(WAX1{k}, m*ra2, []); % rw2*rx1
    tmp = reshape(A{k}, [], m*ra2); % ra1(k)*n    
    WAX1{k} = tmp*WAX1{k}; % size ra1(k)*n, rw2*rx1
    WAX1{k} = reshape(WAX1{k}, [], n*rw2*rx1); % ra1(k)
    WAX1{k} = WAX1{k}.';
    WAX1{k} = reshape(WAX1{k}, n*rw2, []); % rx1*ra1(k)
    WAX1{k} = wc*WAX1{k}; % size rw1, rx1 ra1
    WAX1{k} = reshape(WAX1{k}, rw1*rx1, []); % ra1(k)
    WAX1{k} = WAX1{k}.';
    WAX1{k} = reshape(WAX1{k}, [], rw1, rx1); % ra1(k)
end;
end

% Accumulates the right reduction W{k:d}'*X{k:d}
function [WX1] = rightreduce_vector(WX2, w, x)
% Right WX has the form of the last vector TT block, i.e. [rx, rw]
WX1 = WX2;
[rw1,n,~,rw2] = size(w);
wc = reshape(w, rw1, n*rw2);
Rx = size(x,2);
for k=1:Rx
    [rx1,n,rx2] = size(x{k});
    tmp = reshape(x{k}, rx1*n, rx2);
    WX1{k} = tmp*WX1{k}; % size rx1 n x rw2
    WX1{k} = reshape(WX1{k}, rx1, n*rw2);
    WX1{k} = WX1{k}*wc'; % size rx1, rw1
end;
end

% A matrix-vectors product for the matrix in the 3D TT (WAX1-A-WAX2), and
% full vectors of size (rx1*m*rx2) x b. Returns (rw1*n*rw2) x b
function [w]=local_matvec(x, rx1,m,rx2,b, rw1,n,rw2, WAX1, A, WAX2, Ra,ra1,ra2)
w = zeros(rw1*n*rw2, b);
xc = reshape(x, [], b); % rx1*m*rx2
xc = xc.';
xc = reshape(xc, [], rx2); % b*rx1*m
for k=1:Ra
    tmp = reshape(WAX2{k}, [], rx2); % ra2(k)*rw2
    wk = xc*tmp.';
    wk = reshape(wk, b*rx1, []); % m*ra2(k)*rw2
    wk = wk.';
    wk = reshape(wk, m*ra2(k), []); % rw2*b*rx1
    wk = A{k}*wk;
    wk = reshape(wk, [], rx1); % ra1(k)*n*rw2*b
    wk = wk.';
    wk = reshape(wk, rx1*ra1(k), []); % n*rw2*b
    tmp = reshape(WAX1{k}, rw1, []); % rx1*ra1(k)
    wk = tmp*wk;
    wk = reshape(wk, [], b); % rw1*n*rw2
    w = w+wk;
end;
end


% Builds the full (rw1*n*rw2) x (rx1*m*rx2) matrix from its TT blocks
function [B,sparseflag]=assemble_local_matrix(WAX1, A, WAX2)
Ra = size(A,2);
% Check the sparsity of the matrix blocks
sparseflag = true;
for k=1:Ra
    if (~issparse(A{k}))
        sparseflag=false;
    end;
end;
if (sparseflag)
    B = sparse(0); % B will be in reversed order, r2*r1*n
    for k=1:Ra
        [rw1,rx1,ra1] = size(WAX1{k});
        [ra2,rw2,rx2] = size(WAX2{k});
        Bk = sparse(0);
        if (issparse(A{k}))
            [n,m] = size(A{k});
            n = n/ra1;
            m = m/ra2;            
        else
            [~,n,m,~] = size(A{k});
        end;
        if (ra1>1)
            Ak = reshape(A{k}, ra1, []); % n*m*ra2
            for j=1:ra1
                tmp = WAX1{k}(:,:,j);
                tmp = sparse(tmp);
                Akj = Ak(j,:);
                Akj = sparse(Akj);
                Akj = reshape(Akj, n, m*ra2);
                Bk = Bk + kron(Akj, tmp);
            end;
        else
            tmp = sparse(WAX1{k});
            Ak = sparse(A{k});
            Bk = Bk + kron(Ak, tmp);
        end;
        if (ra2>1)
            Bk = reshape(Bk, rw1*n*rx1*m, ra2);
            for j=1:ra2
                tmp = reshape(WAX2{k}(j,:,:), rw2, rx2);
                tmp = sparse(tmp);
                Ak = Bk(:,j);
                Ak = reshape(Ak, rw1*n, rx1*m);
                B = B + kron(Ak, tmp);
            end;
        else
            tmp = reshape(WAX2{k}, rw2, rx2);
            tmp = sparse(tmp);
            B = B + kron(Bk, tmp);
        end;
    end;
else
    % There are dense blocks, everything is dense, and in the natural index
    % order
    B = 0;
    for k=1:Ra
        [rw1,rx1,ra1] = size(WAX1{k});
        [ra2,rw2,rx2] = size(WAX2{k});
        Bk = reshape(WAX1{k}, rw1*rx1, ra1);
        tmp = reshape(A{k}, ra1, []); % n*m*ra2
        if (issparse(tmp))
            % Don't mix sparse if we are already full
            tmp = full(tmp);
            [n,m] = size(A{k});
            n = n/ra1;
            m = m/ra2;
        else
            [~,n,m,~] = size(A{k});
        end;
        Bk = Bk*tmp;
        Bk = reshape(Bk, rw1, rx1, n, m*ra2);
        Bk = permute(Bk, [1,3,2,4]);
        Bk = reshape(Bk, rw1*n*rx1*m, ra2);
        tmp = reshape(WAX2{k}, ra2, rw2*rx2);
        Bk = Bk*tmp;
        Bk = reshape(Bk, rw1*n, rx1*m, rw2, rx2);        
        Bk = permute(Bk, [1,3,2,4]);
        Bk = reshape(Bk, rw1*n*rw2, rx1*m*rx2);
        B = B+Bk;
    end;
end;
end

% Builds the full (rw1*n*rw2) x 1 vector from its TT blocks
function [w]=assemble_local_vector(WX1, x, WX2)
w = 0;
Rx = size(x,2);
rw1 = size(WX1{1}, 1);
rw2 = size(WX2{1}, 2);
for k=1:Rx
    [rx1,n,~,rx2] = size(x{k});
    wk = reshape(x{k}, rx1, n*rx2);
    wk = WX1{k}*wk;
    wk = reshape(wk, rw1*n, rx2);
    wk = wk*WX2{k};
    wk = reshape(wk, rw1*n*rw2, 1);
    w = w+wk;
end;
end


% Builds the Right Block Jacobi preconditioner: take diag over WAX1
function [P]=assemble_local_rjacobi(WAX1, A, WAX2)
Ra = size(A,2);
rx1 = size(WAX1{1}, 2);
rx2 = size(WAX2{1}, 2);
P = cell(1, rx1);
ind1 = (0:rx1-1)*(rx1+1)+1; % cut the diagonal
for i=1:rx1
    P{i} = 0;
end;
for k=1:Ra
    ra1 = size(WAX1{k},3);
    ra2 = size(WAX2{k},1);
    B1 = reshape(WAX1{k}, rx1*rx1, ra1);
    B1 = B1(ind1, :);
    B1 = B1.';
    tmp = reshape(A{k}, [], ra2); % ra1(k)*n*n
    if (issparse(tmp))
        tmp = full(tmp);
        n = size(A{k},1);
        n = n/ra1;
    else
        n = size(A{k},2);
    end;
    Bk = reshape(WAX2{k}, ra2, rx2*rx2);
    Bk = tmp*Bk;
    Bk = reshape(Bk, ra1*n, n, rx2, rx2);
    Bk = permute(Bk, [1,3,2,4]);
    Bk = reshape(Bk, ra1, n*rx2*n*rx2);
    for i=1:rx1
        P{i} = P{i}+B1(:,i).'*Bk;
    end;
end;
for i=1:rx1
    P{i} = reshape(P{i}, n*rx2, n*rx2);
    P{i} = inv(P{i});
end;
end

% Applies the Jacobi preconditioners to (rx1*m*rx2) x b vectors
function [w]=local_precvec(x, rx1,m,rx2,b, P)
if ((size(P,1)==1)&&(size(P,2)==rx1)) % right prec
    w = zeros(m*rx2, b*rx1);
    xc = reshape(x, rx1, m*rx2*b);
    xc = xc.';
    xc = reshape(xc, m*rx2, b*rx1);
    for i=1:rx1
        for j=1:b
            w(:,j+(i-1)*b) = P{i}*xc(:,j+(i-1)*b);
        end;
    end;
    w = reshape(w, m*rx2*b, rx1);
    w = w.';
    w = reshape(w, rx1*m*rx2, b);
end;
end

