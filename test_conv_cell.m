% %%%%% Convection test

fprintf('-----\n!!! This is the TT-Toolbox-free version with limited verbosity. \n!!! Consider using test_conv with TT-Toolbox\n-----\n\n');

try
    maxNumCompThreads(1);
catch
    % Just skip. Sometimes if you specify -singleCompThread in the command
    % line, MATLAB will fail at maxNumCompThreads with scary, so tell him
    % it's okay.
end;

% Load the matrix and initial state in the {d,R} format
load('dat_conv.mat');
norm_u0 = 32.084841915276819;
tau = 0.2;

% Tensor approximation threshold
tol = 1e-4;

% Options for the tamen
opts = struct;
opts.save2norm = true;
opts.max_full_size = 400;
opts.resid_damp = 1000;
opts.local_iters = 10000;
opts.verb=1;

% Number of time steps
N = 100; % one period
ttimes = zeros(N,1); % For CPU times
err = zeros(N,1); % For second norm
rnk = zeros(N,1); % For tensor ranks

% Initialize tamen:
U = {u0; ones(1,64)}; % u0 x ones(number of Chebyshev points in the first run)
% Go on...
for i=1:N
    tic; 
    [U,opts,~,u]=tamen(U,B,tol,opts,obs);
    ttimes(i)=toc;
    err(i) = norm(u{size(u0,1)}, 'fro')/norm_u0-1;
    % Measure the maximal rank
    for k=1:size(u0,1)
        if (size(U{k},4)>rnk(i))
            rnk(i) = size(U{k},4);
        end;
    end;
    fprintf('====== i=%d, CPU time=%g, d|u|=%3.3e\n%s\n\n', i, ttimes(i), err(i), datestr(clock));
end;

figure(1);
subplot(1,2,1);
plot(tau*(1:N), rnk);
legend('max TT rank');
xlabel('time');
subplot(1,2,2);
plot(tau*(1:N), err);
legend('error in 2nd norm');
xlabel('time');

fprintf('Total CPU time: %g\n', sum(ttimes));
