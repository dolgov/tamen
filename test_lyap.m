% %%%% Lyapunov/Poisson test

try
    maxNumCompThreads(1);
catch
    % Just skip. Sometimes if you specify -singleCompThread in the command
    % line, MATLAB will fail at maxNumCompThreads with scary, so tell him
    % it's okay.
end;

% Spatial grid size
n = 10000;

% Tensor approximation threshold
tol = 1e-5;

% One-dimensional Laplace operator
A1 = spdiags(ones(n,1)*[-1,2,-1], [-1,0,1], n, n);
% 1D identity
I1 = speye(n);

% 2D Laplace operator (A1 x I1 + I1 x A1) in {d,R}
A = {A1, I1;
     I1, A1};

% Right-hand side of all ones
b = cell(2,1);
b{1} = ones(1,n,1); b{2} = ones(1,n,1);

% Opts for amen_solve
opts = struct;
opts.max_full_size=Inf; % It is important to solve all system directly using the sparseness
opts.kickrank = 2; % It is enough in this test

% Solution
tic;
u = amen_solve(A, b, tol, opts);
toc;

% u{i} is in the form r1 x n x 1 x r2. Reshape to more convenient
u{1} = reshape(u{1}, n, []);
u{2} = u{2}.';

% Plot the principal components
plot(u{1});
title('principal components of the solution');
legend toggle;

