tAMEn
=======

Time-dependent AMEn algorithm for solution of the ordinary differential equations
in the tensor train format.
See the paper "Alternating minimal energy approach to ODEs and 
conservation laws in tensor product formats", [[arXiv:1403.8085](http://arxiv.org/abs/1403.8085)].


Contents
-----

- Main high-end routines

 `tamen.m`          Algorithm 1 (tAMEn) from the above paper, adaptive time propagator in the TT format.

 `amen_solve.m`     AMEn for the linear system solution in the TT format. 
                  See [[SIAM J. Sci. Comput., 36(5), A2248](http://epubs.siam.org/doi/10.1137/140953289)], and also 
                  [[TT-Toolbox/amen_solve2](http://github.com/oseledets/TT-Toolbox/blob/master/solve/amen_solve2.m)]

- Help and example files

 `ttdR.m`           An Introduction to the {d,R} cell array storage of tensor trains and tensor chains.

 `test_conv.m`      Test file for the convection example (small space and time grids), TT-Toolbox interface.

 `test_conv_cell.m` Test file for the convection example, cell array storage for compatibility.

 `test_lyap.m`      Test file for the Lyapunov equation, a 10000^2 Poisson problem.

 `test_heat_adap.m` Test file for the heat equation and time adaptivity.


- Auxiliary routines (can be also useful besides this project)

 `chebdiff.m`       Creates Chebyshev differentiation matrix for tamen.

 `cheb2_interpolant.m`       Interpolates between two Chebyshev grids.

 `extract_snapshot.m`      Interpolates the global space-time TT solution from tamen into one time point.

- Technical routines (internal/expert use only)

 `amenany_sweep.m`  Technical routine with a broad range of inputs and outputs. 
                  The AMEn iteration is actually performed here.
                  The reason for a separate file is to shorten the interface routines tamen and amen_solve.

 `grumble_vector.m` Checks a TT vector for consistency. *Technical routine*

 `grumble_matrix.m` Checks a TT matrix for consistency. *Technical routine*

 `orthogonalise_block.m` Perform orthogonalisation in a pair of TT blocks. *Technical routine*

- Data

 `dat_conv.mat`     Precomputed TT data in cell arrays for `test_conv_cell.m`
		

Usage
-----

I. Minimal variant
 1. Download this code (using either GIT or "Download ZIP").
 2. Run the MATLAB.
 3. Change to the tamen directory.
 4. Invoke `test_conv_cell.m` and/or `test_lyap.m`

II. Maximal variant
 1. Download the [[TT-Toolbox](http://github.com/oseledets/TT-Toolbox)] (using either GIT or "Download ZIP").
 2. Download this code (using either GIT or "Download ZIP").
 3. Run the MATLAB.
 4. Change to the TT-Toolbox directory, invoke `setup.m` to initialize paths.
 5. Change to the tamen directory.
 6. Invoke `test_conv.m` and `test_heat_adap.m`

III. Grand maximal variant
 1. Go through the steps in II. Maximal variant.
 2. Compile MEX files in the TT-Toolbox.
 3. Compare `amen_solve.m` from this code and `solve/amen_solve2.m` from the TT-Toolbox, 
    with and without MEX plug-ins in the latter.
 4. Read `ttdR.m`.
 5. Try to refactor `test_lyap.m` for the TT-Toolbox and run it on a laptop.
    Feel the necessity of the {d,R} format when the MATLAB exhausts the memory.


More details
-----

More specific information is located in particular .m files.
Type e.g. `help tamen` in MATLAB.

