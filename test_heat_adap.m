% Test time adaptivity on a heat equation on [0,1]^d, integrated until T=0.1
% Requires TT-Toolbox
try
    maxNumCompThreads(1);
catch
    % Just skip. Sometimes if you specify -singleCompThread in the command
    % line, MATLAB will fail at maxNumCompThreads with scary, so tell him
    % it's okay.
end

L = 8;
d = 5;
nt = 16;
tol = 1e-6;

% Dirichlet-Dirichlet Laplace matrix
A = tt_qlaplace_dd(L*ones(1,d));
A = (-0.1*(2^L+1)^2)*A;
% Initial state of all ones
b = tt_ones(2, d*L);

%%
opts = struct('time_error_damp', 1);
fprintf('Running adaptive tamen...\n\n');
tic;
[U,t] = tamen(tkron(b,tt_ones(nt)), A, tol, opts);
t_adapt = toc;
fprintf('CPU time in the adaptive regime is %g\n', t_adapt);
fprintf('Produced %d time intervals\n\n', numel(t));

%% t contains suboptimal time steps, we can iterate over them explicitly
fprintf('Running a sequence of tamens on these steps...\n');
tic;
tau = t{1}(end);
U2 = tkron(b,tt_ones(nt));
opts.verb = 0;
for i=1:numel(t)
    [U2,t2] = tamen(U2, tau*A, tol, opts);
    % New time step
    if (i<numel(t))
        tau = t{i+1}(end) - t{i}(end);
    end
end
t_seq = toc;
fprintf('CPU time in the sequential regime %g\n', t_seq);
fprintf('Ratio t_adapt/t_seq is %g\n', t_adapt/t_seq);
fprintf('Relative error is %g for tolerance %g\n\n', norm(extract_snapshot(U,t,1) - extract_snapshot(U2,t2,1))/norm(extract_snapshot(U,t,1)), tol);

%% Test a geometrically graded mesh with numel(t) intervals
t_geom = 10.^linspace(log10(t{1}(end)), 0, numel(t));
fprintf('Running a sequence of tamens on numel(t) geometric steps...\n');
tic;
tau = t_geom(1);
U3 = tkron(b,tt_ones(nt));
for i=1:numel(t_geom)
    [U3,t3] = tamen(U3, tau*A, tol, opts);
    % New time step
    if (i<numel(t_geom))
        tau = t_geom(i+1) - t_geom(i);
    end
end
t_g = toc;
fprintf('CPU time for the geometric grid is %g\n', t_g);
fprintf('Ratio t_adapt/t_geom is %g\n', t_adapt/t_g);
fprintf('Relative error is %g for tolerance %g\n\n', norm(extract_snapshot(U,t,1) - extract_snapshot(U3,t3,1))/norm(extract_snapshot(U,t,1)), tol);

figure(1);
semilogy(1:numel(t)-1, diff(cellfun(@(x)x(end), t)), 1:numel(t)-1, diff(t_geom));
title('time steps');
legend('adapt', 'geom');

%% Test a geometrically graded mesh with 100 intervals
t_geom2 = 10.^linspace(-6, 0, 100);
fprintf('Running a sequence of tamens on 100 geometric steps...\n');
tic;
tau = t_geom2(1);
U4 = tkron(b,tt_ones(nt));
for i=1:numel(t_geom2)
    [U4,t4] = tamen(U4, tau*A, tol, opts);
    % New time step
    if (i<numel(t_geom2))
        tau = t_geom2(i+1) - t_geom2(i);
    end
end
t_g2 = toc;
fprintf('CPU time for the a priori geometric grid is %g\n', t_g2);
fprintf('Ratio t_adapt/t_geom_a is %g\n', t_adapt/t_g2);
fprintf('Relative error is %g for tolerance %g\n\n', norm(extract_snapshot(U,t,1) - extract_snapshot(U4,t4,1))/norm(extract_snapshot(U,t,1)), tol);
