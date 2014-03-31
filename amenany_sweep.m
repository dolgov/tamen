% One AMEn iteration for the linear system solution.
%   function [x,rx,z,rz,ZAX,ZY,opts,errs,resids,XAX,XY,XAUX]=amenany_sweep(n, x,rx,A,ra,y,ry,z,rz, tol, opts, ZAX, ZY, aux,raux)
%
% !!! Note !!!!
% This is a technical routine, please use amen_solve or tamen unless you
% understand precisely what are you doing.

function [x,rx,z,rz,ZAX,ZY,opts,errs,resids,XAX,XY,XAUX]=amenany_sweep(n, x,rx,A,ra,y,ry,z,rz, tol, opts, ZAX, ZY, aux,raux)
% XAX: cell(Ra, d+1), XY: cell(Ry, d+1), we may pass XAX(:,i), XAX(:,i+1)
% ZAX: cell(Ra, d+1), ZY: cell(Ry, d+1), analogously.
% XAUX: cell(Raux, d+1) stores XAUX (yours sincerely, Kaptian Obviousity)

% Parse the optional inputs
if (nargin<11)||(isempty(opts))
    opts = struct;
end;
if (nargin<15)
    aux = [];
    raux = [];
end;
if (nargin<12)
    ZAX=[];
end;
if (nargin<13)
    ZY=[];
end;

% Check the inputs for consistency
d = numel(n);
       aas_grumble(x,rx,d,'x');
Ra =   aas_grumble(A,ra,d,'a');
Ry =   aas_grumble(y,ry,d,'y');
Rz =   aas_grumble(z,rz,d,'z');
Raux = aas_grumble(aux,raux,d,'o');

% Adjust the tolerance to the sqrt(d) factor of the recurrent TT-SVD
loctol = tol/sqrt(d);

% Parse opts parameters. We just populate what we do not have by defaults
if (~isfield(opts, 'resid_damp'));      opts.resid_damp=2;      end;
if (~isfield(opts, 'rmax'));            opts.rmax=Inf;          end;
if (~isfield(opts, 'max_full_size'));   opts.max_full_size=50;  end;
if (~isfield(opts, 'local_iters'));     opts.local_iters=100;   end;
if (~isfield(opts, 'trunc_norm'));      opts.trunc_norm='fro';  end;
if (~isfield(opts, 'kickrank2'));       opts.kickrank2=0;       end;
if (~isfield(opts, 'verb'));            opts.verb=1;            end;
if (~isfield(opts, 'local_prec'));      opts.local_prec='';     end;

% More parsing: check what inputs we have, and what reductions we need
if (Ra>0)
    XAX = cell(Ra,d+1);
    for k=1:Ra
        XAX{k,1}=1;
        XAX{k,d+1}=1;
    end;
end;
if (Ry>0)
    XY = cell(Ry,d+1);
    for k=1:Ry
        XY{k,1}=1;
        XY{k,d+1}=1;
    end;
end;
if (Raux>0)
    XAUX = cell(Raux,d+1);
    for k=1:Raux
        XAUX{k,1}=1;
        XAUX{k,d+1}=1;
    end;
end;

% Create ZAX, ZY if they were not passed
if (isempty(ZAX))
    ZAX = cell(Ra, d+1);
    for k=1:Ra
        ZAX{k,1}=1;
        ZAX{k,d+1}=1;
    end;
end;
if (isempty(ZY))
    ZY = cell(Ry+1, d+1);
    for k=1:Ry+1
        ZY{k,1}=1;
        ZY{k,d+1}=1;
    end;
end;

% Right to left iteration: warmup
for i=d:-1:2
    if (~isempty(ZAX{1,i}))
        % ZAX was passed.
        % We assume that X is left-orthogonal, so we may compute the update for Z
        if (Rz>0)
            % Project the residual to the Z-block:
            %   y
            crzy = assemble_local_vector(ZY(:,i), y(i,:), ZY(:,i+1), Ry,ry(i,:),ry(i+1,:), rz(i),n(i),rz(i+1));
            %   Ax
            crzAx = local_matvec(x{i}, rx(i),n(i),rx(i+1),1, rz(i),n(i),rz(i+1), ZAX(:,i), A(i,:), ZAX(:,i+1), Ra,ra(i,:),ra(i+1,:));
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
            [crz,~]=qr(crz.', 0);
            rz(i) = size(crz,2);
            z{i} = crz.';
        end;
    else
        % No information is given, just orthogonalize the residual blocks
        crz = reshape(z{i}, rz(i), n(i)*rz(i+1));
        [crz, rv]=qr(crz.', 0);
        cr2 = z{i-1};
        cr2 = reshape(cr2, rz(i-1)*n(i-1), rz(i));
        cr2 = cr2*rv.';
        rz(i) = size(crz, 2);
        crz = reshape(crz.', rz(i), n(i), rz(i+1));
        z{i-1} = reshape(cr2, rz(i-1), n(i-1), rz(i));
        z{i} = crz;
    end;
    
    % Orthogonalization for X
    crx = reshape(x{i}, rx(i), n(i)*rx(i+1));
    [crx, rv]=qr(crx.', 0);
    cr2 = x{i-1};
    cr2 = reshape(cr2, rx(i-1)*n(i-1), rx(i));
    cr2 = cr2*rv.';
    rx(i) = size(crx, 2);
    crx = reshape(crx.', rx(i), n(i), rx(i+1));
    x{i-1} = reshape(cr2, rx(i-1), n(i-1), rx(i));
    x{i} = crx;
    
    % Compute reductions
    % With X
    XAX(:,i) = rightreduce_matrix(XAX(:,i+1), crx, A(i,:), crx, rx(i),n(i),rx(i+1), Ra,ra(i,:),ra(i+1,:), rx(i),n(i),rx(i+1));
    XY(:,i) = rightreduce_vector(XY(:,i+1), crx, y(i,:), rx(i),n(i),rx(i+1), Ry,ry(i,:),ry(i+1,:));
    % With Z
    ZAX(:,i) = rightreduce_matrix(ZAX(:,i+1), z{i}, A(i,:), crx, rz(i),n(i),rz(i+1), Ra,ra(i,:),ra(i+1,:), rx(i),n(i),rx(i+1));
    ZY(:,i) = rightreduce_vector(ZY(:,i+1), z{i}, y(i,:), rz(i),n(i),rz(i+1), Ry,ry(i,:),ry(i+1,:));
end;

% Initialize the error and residual storages
errs = zeros(d,1);
resids = zeros(d,1);

% Left to right AMEn iteration: solution
for i=1:d
    % Prepare the local system
    rhs = assemble_local_vector(XY(:,i), y(i,:), XY(:,i+1), Ry,ry(i,:),ry(i+1,:), rx(i),n(i),rx(i+1));
    sol_prev = reshape(x{i}, rx(i)*n(i)*rx(i+1),1);
    norm_rhs = norm(rhs);
    % Extract the matrix parts, accelerate a plenty of iterations with them
    XAX1 = XAX(:,i);
    Ai = A(i,:);
    XAX2 = XAX(:,i+1);
    ra1 = ra(i,:);
    ra2 = ra(i+1,:);
    % Measure the residual
    resids(i) = norm(rhs-local_matvec(sol_prev, rx(i),n(i),rx(i+1),1, rx(i),n(i),rx(i+1), XAX1, Ai, XAX2, Ra,ra1,ra2))/norm_rhs;
    
    if (rx(i)*n(i)*rx(i+1)<opts.max_full_size)
        % If the system size is small, assemble the full system and solve
        % directly
        [B,sparseflag] = assemble_local_matrix(XAX1, Ai, XAX2, Ra,ra1,ra2, rx(i),n(i),rx(i+1), rx(i),n(i),rx(i+1));
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
        lociters = opts.local_iters;
        % Extract the current number of iterations wisely
        if (numel(lociters)>1)
            lociters = lociters(i);
        end;
        % Run the bicgstab without a preconditioner first
        [sol,flg,relres,iter] = bicgstab(@(x)local_matvec(x, rx(i),n(i),rx(i+1),1, rx(i),n(i),rx(i+1), XAX1, Ai, XAX2, Ra,ra1,ra2),...
            rhs, max(loctol/opts.resid_damp,eps*2), lociters, [], [], sol_prev);
        % Report
        if ((opts.verb>1)&&(flg==0))
            fprintf('\tamen_sweep: i=%d. Bicgstab: %g iters, residual %3.3e. ', i, iter, relres);
        end;
        % If we did not converge...
        if (flg>0)&&(lociters>1)
            precfun = [];
            % ... use the Block Jacobi preconditioner, if required            
            if (strcmp(opts.local_prec, 'r'))
                P = assemble_local_rjacobi(XAX1, Ai, XAX2, Ra,ra1,ra2, rx(i),n(i),rx(i+1));
                precfun = @(x)local_precvec(x, rx(i),n(i),rx(i+1),1, P);
            end;
            % In any case, run the bicg once again
            [sol,~,relres,iter] = bicgstab(@(x)local_matvec(x, rx(i),n(i),rx(i+1),1, rx(i),n(i),rx(i+1), XAX1, Ai, XAX2, Ra,ra1,ra2),...
                rhs, max(loctol/opts.resid_damp,eps*2), lociters, precfun, [], sol);
            % And report its performance
            if (opts.verb>1)
                fprintf('\tamen_sweep: i=%d. Bicgstab(prec): %g iters, residual %3.3e. ', i, lociters+iter, relres);
            end;
        end;
    end;
    
    % Measure the error
    errs(i) = norm(sol-sol_prev)/norm(sol);
    
    % Truncation
    sol = reshape(sol, rx(i)*n(i), rx(i+1));
    if (loctol>0)&&(i<d)
        [u,s,v]=svd(sol,'econ');
        s = diag(s);
        if (strcmp(opts.trunc_norm, 'fro'))
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
            res_new = norm(rhs-local_matvec(sol, rx(i),n(i),rx(i+1),1, rx(i),n(i),rx(i+1), XAX1, Ai, XAX2, Ra,ra1,ra2))/norm_rhs;
            % Start from the previous rank...
            r = min(rx(i)*n(i),rx(i+1));
            r = max(r-rz(i+1,1), 1);
            cursol = u(:,1:r)*diag(s(1:r))*v(:,1:r)';
            % ... check the corresponding residual
            res = norm(rhs-local_matvec(cursol, rx(i),n(i),rx(i+1),1, rx(i),n(i),rx(i+1), XAX1, Ai, XAX2, Ra,ra1,ra2))/norm_rhs;
            if (res<max(loctol, res_new*opts.resid_damp))
                drank = -1; % rank is overestimated, decrease it
            else
                drank = 1; % residual is large; increase the rank
            end;
            while (r>0)&&(r<=numel(s))
                % Pick the ranks in the selected direction one by one
                cursol = u(:,1:r)*diag(s(1:r))*v(:,1:r)';
                res = norm(rhs-local_matvec(cursol, rx(i),n(i),rx(i+1),1, rx(i),n(i),rx(i+1), XAX1, Ai, XAX2, Ra,ra1,ra2))/norm_rhs;
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
        crzy = assemble_local_vector(ZY(:,i), y(i,:), ZY(:,i+1), Ry,ry(i,:),ry(i+1,:), rz(i),n(i),rz(i+1));
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
        [crz,~]=qr(crz, 0);
        % Careful: store the old rank of z, since it is that will be used
        % in the solution enrichment, not the updated value after the QR
        rzold = rz(i+1);
        rz(i+1) = size(crz,2); % Now replace it
        z{i} = crz;
    end;
    
    if (i<d)
        % Apply enrichment for the solution
        if (Rz>0)
            % y
            crzy = assemble_local_vector(XY(:,i), y(i,:), ZY(:,i+1), Ry,ry(i,:),ry(i+1,:), rx(i),n(i),rzold);
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
                XAUX{k,i+1} = rv(:,1:raux(i+1,k));
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
        x{i+1} = cr2; % now it is a good initial guess
        x{i} = reshape(u, rx(i), n(i), rx(i+1));
        
        % Update reductions
        % For X
        XAX(:,i+1) = leftreduce_matrix(XAX(:,i), u, Ai, u, rx(i),n(i),rx(i+1), Ra,ra1,ra2, rx(i),n(i),rx(i+1));
        XY(:,i+1) = leftreduce_vector(XY(:,i), u, y(i,:), rx(i),n(i),rx(i+1), Ry,ry(i,:),ry(i+1,:));
        % For Z
        ZAX(:,i+1) = leftreduce_matrix(ZAX(:,i), z{i}, Ai, u, rz(i),n(i),rz(i+1), Ra,ra1,ra2, rx(i),n(i),rx(i+1));
        ZY(:,i+1) = leftreduce_vector(ZY(:,i), z{i}, y(i,:), rz(i),n(i),rz(i+1), Ry,ry(i,:),ry(i+1,:));
    else
        x{i} = reshape(sol, rx(i), n(i), rx(i+1));
    end;
end;

end

% Accumulates the left reduction W{1:k}'*A{1:k}*X{1:k}
function [WAX2] = leftreduce_matrix(WAX1, w, A, x, rw1,n,rw2, Ra,ra1,ra2, rx1,m,rx2)
% Left WAX has the form of the first matrix TT block, i.e. [rw, rx, ra]
WAX2 = WAX1;
wc = reshape(w, rw1, n*rw2);
xc = reshape(x, rx1*m, rx2);
for k=1:Ra
    WAX2{k} = reshape(WAX2{k}, rw1, rx1*ra1(k));
    WAX2{k} = wc'*WAX2{k}; % size n rw2 x rx1 ra1
    WAX2{k} = reshape(WAX2{k}, n, rw2*rx1*ra1(k));
    WAX2{k} = WAX2{k}.';
    WAX2{k} = reshape(WAX2{k}, rw2*rx1, ra1(k)*n);
    tmp = reshape(A{k}, ra1(k)*n, m*ra2(k));
    WAX2{k} = WAX2{k}*tmp; % size rw2 rx1 m ra2
    WAX2{k} = reshape(WAX2{k}, rw2, rx1*m*ra2(k));
    WAX2{k} = WAX2{k}.';
    WAX2{k} = reshape(WAX2{k}, rx1*m, ra2(k)*rw2);
    WAX2{k} = xc.'*WAX2{k}; % size rx2, ra2 rw2
    WAX2{k} = reshape(WAX2{k}, rx2*ra2(k), rw2);
    WAX2{k} = WAX2{k}.';
end;
end

% Accumulates the left reduction W{1:k}'*X{1:k}
function [WX2] = leftreduce_vector(WX1, w, x, rw1,n,rw2, Rx,rx1,rx2)
% Left WX has the form of the first vector TT block, i.e. [rw, rx]
WX2 = WX1;
wc = reshape(w, rw1, n*rw2);
for k=1:Rx
    WX2{k} = wc'*WX2{k}; % size n rw2 x rx1
    WX2{k} = reshape(WX2{k}, n, rw2*rx1(k));
    WX2{k} = WX2{k}.';
    WX2{k} = reshape(WX2{k}, rw2, rx1(k)*n);
    tmp = reshape(x{k}, rx1(k)*n, rx2(k));
    WX2{k} = WX2{k}*tmp; % size rw2, rx2
end;
end

% Accumulates the right reduction W{k:d}'*A{k:d}*X{k:d}
function [WAX1] = rightreduce_matrix(WAX2, w, A, x, rw1,n,rw2, Ra,ra1,ra2, rx1,m,rx2)
% Right WAX has the form of the last matrix TT block, i.e. [ra, rw, rx]
WAX1 = WAX2;
wc = reshape(w, rw1, n*rw2);
wc = conj(wc);
xc = reshape(x, rx1*m, rx2);
for k=1:Ra
    WAX1{k} = reshape(WAX1{k}, ra2(k)*rw2, rx2);
    WAX1{k} = xc*WAX1{k}.'; % size rx1 m x ra2 rw2
    WAX1{k} = reshape(WAX1{k}, rx1, m*ra2(k)*rw2);
    WAX1{k} = WAX1{k}.';
    WAX1{k} = reshape(WAX1{k}, m*ra2(k), rw2*rx1);
    tmp = reshape(A{k}, ra1(k)*n, m*ra2(k));
    WAX1{k} = tmp*WAX1{k}; % size ra1(k)*n, rw2*rx1
    WAX1{k} = reshape(WAX1{k}, ra1(k), n*rw2*rx1);
    WAX1{k} = WAX1{k}.';
    WAX1{k} = reshape(WAX1{k}, n*rw2, rx1*ra1(k));
    WAX1{k} = wc*WAX1{k}; % size rw1, rx1 ra1
    WAX1{k} = reshape(WAX1{k}, rw1*rx1, ra1(k));
    WAX1{k} = WAX1{k}.';
end;
end

% Accumulates the right reduction W{k:d}'*X{k:d}
function [WX1] = rightreduce_vector(WX2, w, x, rw1,n,rw2, Rx,rx1,rx2)
% Right WX has the form of the last vector TT block, i.e. [rx, rw]
WX1 = WX2;
wc = reshape(w, rw1, n*rw2);
for k=1:Rx
    tmp = reshape(x{k}, rx1(k)*n, rx2(k));
    WX1{k} = tmp*WX1{k}; % size rx1 n x rw2
    WX1{k} = reshape(WX1{k}, rx1(k), n*rw2);
    WX1{k} = WX1{k}*wc'; % size rx1, rw1
end;
end

% A matrix-vectors product for the matrix in the 3D TT (WAX1-A-WAX2), and
% full vectors of size (rx1*m*rx2) x b. Returns (rw1*n*rw2) x b
function [w]=local_matvec(x, rx1,m,rx2,b, rw1,n,rw2, WAX1, A, WAX2, Ra,ra1,ra2)
w = zeros(rw1*n*rw2, b);
xc = reshape(x, rx1*m*rx2, b);
xc = xc.';
xc = reshape(xc, b*rx1*m, rx2);
for k=1:Ra
    tmp = reshape(WAX2{k}, ra2(k)*rw2, rx2);
    wk = xc*tmp.';
    wk = reshape(wk, b*rx1, m*ra2(k)*rw2);
    wk = wk.';
    wk = reshape(wk, m*ra2(k), rw2*b*rx1);
    tmp = reshape(A{k}, ra1(k)*n, m*ra2(k));
    wk = tmp*wk;
    wk = reshape(wk, ra1(k)*n*rw2*b, rx1);
    wk = wk.';
    wk = reshape(wk, rx1*ra1(k), n*rw2*b);
    tmp = reshape(WAX1{k}, rw1, rx1*ra1(k));
    wk = tmp*wk;
    wk = reshape(wk, rw1*n*rw2, b);
    w = w+wk;
end;
end

% Builds the full (rw1*n*rw2) x (rx1*m*rx2) matrix from its TT blocks
function [B,sparseflag]=assemble_local_matrix(WAX1, A, WAX2, Ra,ra1,ra2, rw1,n,rw2, rx1,m,rx2)
% Check the sparsity of the matrix blocks
sparseflag = true;
for k=1:Ra
    if (~issparse(A{k}))
        sparseflag=false;
    end;
end;
if (sparseflag)
    B = sparse(rw2*rw1*n, rx2*rx1*m); % reverse order !!!
    % The reverse order is needed since usually the n x m part is large and
    % sparse, so let it be the senior dimension.
    % Note that currently only canonical sparse matrices are allowed
    for k=1:Ra
        tmp = reshape(WAX2{k}, rw2, rx2);
        tmp = sparse(tmp);
        Bk = reshape(WAX1{k}, rw1, rx1);
        Bk = sparse(Bk);
        Bk = kron(Bk, tmp); % mind endiannes
        Bk = kron(A{k}, Bk); % mind endiannes
        B = B+Bk;
    end;
else
    % There are dense blocks, everything is dense, and in the natural index
    % order
    B = zeros(rw1*n*rw2, rx1*m*rx2);
    for k=1:Ra
        Bk = reshape(WAX1{k}, rw1*rx1, ra1(k));
        tmp = reshape(A{k}, ra1(k), n*m*ra2(k));
        if (issparse(tmp))
            % Don't mix sparse if we are already full
            tmp = full(tmp);
        end;
        Bk = Bk*tmp;
        Bk = reshape(Bk, rw1, rx1, n, m*ra2(k));
        Bk = permute(Bk, [1,3,2,4]);
        Bk = reshape(Bk, rw1*n*rx1*m, ra2(k));
        tmp = reshape(WAX2{k}, ra2(k), rw2*rx2);
        Bk = Bk*tmp;
        Bk = reshape(Bk, rw1*n, rx1*m, rw2, rx2);
        Bk = permute(Bk, [1,3,2,4]);
        Bk = reshape(Bk, rw1*n*rw2, rx1*m*rx2);
        B = B+Bk;
    end;
end;
end

% Builds the full (rw1*n*rw2) x 1 vector from its TT blocks
function [w]=assemble_local_vector(WX1, x, WX2, Rx,rx1,rx2, rw1,n,rw2)
w = zeros(rw1*n*rw2, 1);
for k=1:Rx
    wk = reshape(x{k}, rx1(k), n*rx2(k));
    wk = WX1{k}*wk;
    wk = reshape(wk, rw1*n, rx2(k));
    wk = wk*WX2{k};
    wk = reshape(wk, rw1*n*rw2, 1);
    w = w+wk;
end;
end

% Builds the Right Block Jacobi preconditioner: take diag over WAX1
function [P]=assemble_local_rjacobi(WAX1, A, WAX2, Ra,ra1,ra2, rx1,n,rx2)
P = cell(1, rx1);
ind1 = (0:rx1-1)*(rx1+1)+1; % cut the diagonal
for i=1:rx1
    P{i} = zeros(1, n*rx2*n*rx2);
end;
for k=1:Ra
    B1 = reshape(WAX1{k}, rx1*rx1, ra1(k));
    B1 = B1(ind1, :);
    B1 = B1.';
    tmp = reshape(A{k}, ra1(k)*n*n, ra2(k));
    if (issparse(tmp))
        tmp = full(tmp);
    end;
    Bk = reshape(WAX2{k}, ra2(k), rx2*rx2);
    Bk = tmp*Bk;
    Bk = reshape(Bk, ra1(k)*n, n, rx2, rx2);
    Bk = permute(Bk, [1,3,2,4]);
    Bk = reshape(Bk, ra1(k), n*rx2*n*rx2);
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

% Checks the inputs for basic consistency
function [R]=aas_grumble(x,rx,d,objtype)
if (isempty(x))
    R = 0; % This shows that this input is absent
else
    R = size(x,2);
    if ((strcmp(objtype, 'x')||strcmp(objtype, 'z'))&&(R>1))
        error('Tensor Chain format (R>1) is not allowed for the input %s', objtype);
    end;
    if (size(rx,2)~=R)
        error('Inconsistent canonical ranks in the input %s', objtype);
    end;
    if ((size(x,1)~=d)||(size(rx,1)~=(d+1)))
        error('Inconsistent dimensions of the input %s', objtype);
    end;
end;
end
