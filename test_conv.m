% %%%%% Convection test

try
    maxNumCompThreads(1);
catch
    % Just skip. Sometimes if you specify -singleCompThread in the command
    % line, MATLAB will fail at maxNumCompThreads with scary, so tell him
    % it's okay.
end;

% One-dimension spatial grid size
n = 2*ones(9,1);
h = 20/prod(n);

% Uniform grid points
x = tt_x(n)*h-10;

% Tensor approximation threshold
tol = 1e-4;

% Construct the central difference matrix in 1D
A = tt_shift(n,numel(n),-1)-tt_shift(n,numel(n),1);
A = A - tt_matrix(tt_unit(n,numel(n),1),n,1)*tt_matrix(tt_unit(n,numel(n),n),1,n);
A = A + tt_matrix(tt_unit(n,numel(n),n),n,1)*tt_matrix(tt_unit(n,numel(n),1),1,n);
A = A*(0.5/h);

% The central difference matrix in 2D
B = tkron(A, tt_eye(n))+tkron(tt_eye(n),A);

% Prepare the initial state u0=exp(-x^2)
u0 = amen_cross({x}, @(x)exp(-x.^2), 1e-14, 'y0', tt_unit(n,numel(n),prod(n)/2));

% Make the two-dimensional initial state exp(-x^2-y^2)
u0 = tkron(u0,u0);

% Options for the tamen
opts = struct;
opts.save2norm = true;
opts.max_full_size = 400;
opts.resid_damp = 1000;
opts.local_iters = 10000;
opts.verb=1;

% Linear invariant is the sum of the elements == dot(ones,u)
obs = {tt_ones([n;n])};

% Single time interval
tau = 0.2;
B = B*tau;

% Number of time steps
N = 100; % one period
ttimes = zeros(N,1); % For CPU times
err = zeros(N,3); % d<o|u>, d|u|, |u-u0|
rnk = zeros(N,1); % For tensor ranks

% Initialize tamen:
U = tkron(u0,tt_ones(64)); % u0 x ones(number of Chebyshev points in the first run)
% Go on...
for i=1:N
    tic; 
    [U,opts,~,u]=tamen(U,B,tol,opts,obs);
    ttimes(i)=toc;
    rnk(i) = max(U.r);
    err(i,1) = dot(obs{1}, u)/dot(obs{1},u0)-1; % Error in the 1st norm
    err(i,2) = norm(u)/norm(u0)-1; % Error in the 2nd norm
    err(i,3) = norm(u-u0)/norm(u0); % Discrepancy with u0. Must be small in the final step
    fprintf('====== i=%d, CPU time=%g, d<o|u>=%3.3e, d|u|=%3.3e, |u-u0|=%3.3e\n%s\n\n', i, ttimes(i), err(i,1), err(i,2), err(i,3), datestr(clock));
end;

figure(1);
subplot(1,2,1);
plot(tau*(1:N), rnk);
legend('max TT rank');
xlabel('time');
subplot(1,2,2);
plot(tau*(1:N), err(:,1), tau*(1:N), err(:,2));
legend('error in 1st norm', 'error in 2nd norm');
xlabel('time');

fprintf('Total CPU time: %g\n', sum(ttimes));
