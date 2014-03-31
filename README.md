tAMEn
=======

Time-dependent AMEn algorithm for solution of the ordinary differential equations
in the tensor train format.
See the paper "Alternating minimal energy approach to ODEs and 
conservation laws in tensor product formats", [[arXiv:](http://arxiv.org/abs/)].


Contents
-----
tamen.m		 Algorithm 1 (tAMEn) from the above paper, time-dependent propagator in the TT format.
amen_solve.m	 AMEn for the linear system solution in the TT format. 
		 See [[arXiv:1304.1222](http://arxiv.org/abs/1304.1222)], [[arXiv:1301.6068](http://arxiv.org/abs/1301.6068)], and also 
		 [[TT-Toolbox/amen_solve2](http://github.com/oseledets/TT-Toolbox/blob/master/solve/amen_solve2.m)]
ttdR.m		 An introduction to the {d,R} cell array storage of tensor trains and tensor chains.
test_conv.m	 Test file for the convection example (small space and time grids), TT-Toolbox interface.
test_conv_cell.m Test file for the convection example, cell array storage for compatibility.
test_lyap.m	 Test file for the Lyapunov equation, a 10000^2 Poisson problem.
amenany_sweep.m	 Technical routine with a broad range of inputs and outputs. 
		 The AMEn iteration is actually performed here.
		 The reason for a separate file is to shorten the interface routines tamen and amen_solve
dat_conv.mat	 Precomputed TT data in cell arrays for test_conv_cell.m
		

Usage
-----

I. Minimal variant
	1. Download this code (using either GIT or "Download ZIP").
	2. Run the MATLAB.
	3. Change to the tamen directory.
	4. Invoke `test_conv_cell.m` and/or `test_lyap.m`

II. Maximal variant
	1. Download the TT-Toolbox from [http://github.com/oseledets/TT-Toolbox] (using either GIT or "Download ZIP").
	2. Download this code (using either GIT or "Download ZIP").
	3. Run the MATLAB.
	4. Change to the TT-Toolbox directory, invoke `setup.m` to initialize paths.
	5. Change to the tamen directory.
	6. Invoke `test_conv.m`

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

