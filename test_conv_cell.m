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
load('dat_conv.mat'); % read u0, B, obs
norm_u0 = 64.1696838305535948;
mass_u0 = 8235.4966458264279936;
tau = 0.2;

% Tensor approximation threshold
tol = 1e-4;

% Options for the tamen
opts = struct;
% we want to solve local systems accurately
% since the ranks are small, use the direct solver always
opts.max_full_size = inf;

% Number of time steps
N = 100; % one period
ttimes = zeros(N,1); % For CPU times
err = zeros(N,2); % For firt and second norms
rnk = zeros(N,1); % For tensor ranks

% Initialize tamen:
U = [u0; {ones(1,16)}]; % u0 x ones(number of Chebyshev points in the first run)
% Go on...
for i=1:N
    % Run tamen for the current time interval
    tic; 
    [U,t,opts]=tamen(U,B,tol,opts,obs);
    ttimes(i)=toc;
    % Extract u at the end of the interval
    u = extract_snapshot(U,t,1);
    % Compute the sum
    mass_u = 1;
    for j=1:size(u,1)
        mass_u = mass_u*reshape(sum(u{j},2), size(mass_u,2), []);
    end;
    err(i,1) = mass_u/mass_u0-1;
    % Compute the second norm
    err(i,2) = norm(u{size(u0,1)}, 'fro')/norm_u0-1;
    % Measure the maximal rank
    for k=1:size(u0,1)
        if (size(u{k},4)>rnk(i))
            rnk(i) = size(u{k},4);
        end;
    end;
    fprintf('====== i=%d, CPU time=%g, rank=%d, d<o|u>=%3.3e, d|u|=%3.3e\n', i, ttimes(i), rnk(i), err(i,1), err(i,2));
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
