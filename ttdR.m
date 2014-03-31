% {d,R} TT representation scheme     ****This is a help file only
%
% The so-called {d,R} TT representation is introduced to cope with
% sparseness and store CP and TC formats transparently. The linear storage
% scheme employed in TT-Toolbox does not really admit sparse TT blocks of
% matrices: extraction of a huge chunk from a sparse vector is expensive,
% moreover, it cannot be reshaped to a r1 x n x m x r2 TT block.
% Besides, in many problems the matrix can be written as a sum of tensor
% trains (possibly with rank-1 terms), but its compression is prohibited,
% from both the accuracy (a small frobenius-norm error in the matrix may
% yield a large error in its spectral components) and efficiency (how to
% compress sparse blocks?) reasons.
%
% To get rid of these problems, we have started a slow transition back to the
% flavour of the TT1.0 toolbox, i.e. the cell array storage of TT blocks.
% The memory overhead is in most cases negligible, but the flexibility is 
% greatly improved. The {d,R} rule writes as follows:
%   Take strong Kronecker products over the first cell dimension, and
%   sum over the second cell dimension.
% In other terms, a single d-dimensional tensor train writes as a d x 1
% cell array of d TT blocks. A sum of two TTs (without compression) becomes 
% a simple concatenation into a d x 2 cell array, and so on.
% The two-dimensional Laplace operator, for example, may be written as 
% a 2 x 2 cell array, with 1D Laplacians stored in {1,1} and {2,2}, and
% identities put into {2,1} and {1,2}. Thus, the sparseness, or (in future)
% toeplitzness incorporate transparently.
%
% To handle TT matrices and TT vectors in the same manner, the common
% dimension order is introduced for each block {i,k}.
% If the block is dense double, it must be a four-dimensional array of the
% form r1 x n x m x r2, for both matrix and vector. In the vector case,
% either n or m can be 1. If the block is sparse, it is always a n x m
% matrix. Currently, this means that a sparse matrix must be always written
% in the CP form, i.e. as the sum of rank-1 TT matrices, since there is no
% way to detect larger TT ranks, and no way to reshape sparse matrices to
% higher-dimensional arrays.

