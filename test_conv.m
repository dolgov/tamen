% %%%%% Convection test

try
    maxNumCompThreads(1);
catch
    % Just skip. Sometimes if you specify -singleCompThread in the command
    % line, MATLAB will fail at maxNumCompThreads with scary, so tell him
    % it's okay.
end;

% One-dimension spatial grid size
n = 2*ones(12,1);
h = 20/prod(n);

% Uniform grid points
x = tt_x(n)*h-10;

% Tensor approximation threshold
tol = 1e-5;

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
% we want to solve local systems accurately
% since the ranks are small, use the direct solver always
opts.max_full_size = inf;
opts.time_scheme = 'cheb';
opts.verb = 1; % we have our own fprintf

% Linear invariant is the sum of the elements == dot(ones,u)
obs = {tt_ones([n;n])};

% Single time interval
tau = 0.2;
B = B*tau;

% Number of time steps
N = 500; % one period is 100(*0.2)
ttimes = zeros(N,1); % For CPU times
err = zeros(N,3); % d<o|u>, d|u|, |u-u0|
rnk = zeros(N,1); % For tensor ranks

% Initialize tamen:
U = tkron(u0,tt_ones(8)); % u0 x ones(number of time points)
% Go on...
for i=1:N
    tic; 
    [U,~,opts,u]=tamen(U,B,tol,opts,obs);
    ttimes(i)=toc;
    rnk(i) = max(u.r);
    err(i,1) = dot(obs{1}, u)/dot(obs{1},u0)-1; % Error in the sum
    err(i,2) = norm(u)/norm(u0)-1; % Error in the 2nd norm
    err(i,3) = norm(u-u0)/norm(u0); % Discrepancy with u0. Must be small in the final step
    fprintf('====== i=%d, CPU time=%g, rank=%d, d<o|u>=%3.3e, d|u|=%3.3e, |u-u0|=%3.3e\n', i, ttimes(i), rnk(i), err(i,1), err(i,2), err(i,3));
end;

figure(1);
subplot(1,2,1);
plot(tau*(1:N), rnk);
legend('max TT rank');
xlabel('time');
subplot(1,2,2);
plot(tau*(1:N), err(:,1), tau*(1:N), err(:,2));
legend('error in sum', 'error in 2nd norm');
xlabel('time');

fprintf('Total CPU time: %g\n', sum(ttimes));
