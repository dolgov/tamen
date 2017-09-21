% Check for consistency and retrieve the sizes of a matrix in various TT formats
%   function [d,n,Ra,ra]=grumble_matrix(a,aname,d,n,Ra,ra)
%
% !!! Note !!!!
% This is a technical routine, please use higher-level amen_solve or tamen 
% unless you understand precisely what are you doing.

function [d,n,Ra,ra]=grumble_matrix(a,aname,d,n,Ra,ra)
if (isempty(a))
    d = 0;
    n = 0;
    Ra = 0;
    ra = 0;
    return;
end;
if (isa(a, 'tt_matrix'))
    if (nargin>2)&&(~isempty(d))&&(a.d~=d)
        error('dim of %s is inconsistent',aname);
    else
        d = a.d;
    end;
    if (nargin>3)&&(~isempty(n))&&((numel(a.n)~=numel(n))||(~all(a.n==n))||(~all(a.m==n)))
        error('mode sizes of %s are inconsistent', aname);
    else
        n = a.n;
    end;
    if (nargin>4)&&(~isempty(Ra))&&(Ra~=1)
        error('canonical rank of %s is inconsistent', aname);
    else
        Ra=1;
    end;
    if (nargin>5)&&(~isempty(ra))&&((numel(a.r)~=numel(ra))||(~all(a.r==ra)))
        error('TT ranks of %s are inconsistent', aname);
    else
        ra = a.r;
    end;
else
    % {d,R} format
    if (nargin>2)&&(~isempty(d))&&(size(a,1)~=d)
        error('dim of %s is inconsistent',aname);
    else
        d = size(a,1);
    end;
    if (nargin>4)&&(~isempty(Ra))&&(Ra~=size(a,2))
        error('canonical rank of %s is inconsistent', aname);
    else
        Ra = size(a,2);
    end;
    n_in = ones(d,1);
    if (nargin>3)&&(~isempty(n))
        if (numel(n)~=d)
            error('mode sizes of %s are inconsistent', aname);
        end;
    end;
    if (nargin<=5)||(isempty(ra))
        ra_in = ones(d+1,Ra);
    else
        if (size(ra,1)~=(d+1))||(size(ra,2)~=Ra)
            error('TT ranks of %s are inconsistent', aname);
        end;
    end;
    for k=1:Ra
        for i=1:d
            if (issparse(a{i,k}))
                % Sparse block, the sizes should be r1*n x n*r2
                [n1,n2]=size(a{i,k});
                if (nargin>3)&&(~isempty(n))
                    ri = n1/n(i);
                    ri2 = n2/n(i);
                    if (abs(ri-round(ri))>sqrt(eps))||(abs(ri2-round(ri2))>sqrt(eps))
                        error('%s{%d,%d} is sparse, but the sizes are not divisible by n', aname, i, k);
                    end;
                    if (nargin>5)&&(~isempty(ra))&&((ra(i+1,k)~=ri2)||(ra(i,k)~=ri))
                        error('TT rank (%d,%d) of %s is inconsistent', i, k, aname);
                    end;
                    ra_in(i,k) = ri;
                    ra_in(i+1,k) = ri2;
                    n_in(i) = n(i);
                else
                    % no n is given, the only way to extract it is to
                    % assume a rank-1 block
                    n_in(i) = size(a{i,k},1);
                    if (size(a{i,k},2)~=n_in(i)) % Square matrix only for now
                        error('block (%d,%d) of %s is not square, or a matrix is of TT rank >1', i, k, aname);
                    end;
                    ra_in(i,k) = 1;
                    ra_in(i+1,k) = 1;
                    if (nargin>5)&&(~isempty(ra))&&((ra(i+1,k)~=ra_in(i,k))||(ra(i,k)~=ra_in(i+1,k)))
                        error('TT rank (%d,%d) of %s is inconsistent', i, k, aname);
                    end;                    
                end;                                
            else
                % Dense 4-dimensional blocks
                n_in(i) = size(a{i,k},2);
                if (size(a{i,k},3)~=n_in(i)) % Square matrix only for now
                    error('mode size (%d,%d) of %s is inconsistent', i, k, aname);
                end;
                if (nargin>3)&&(~isempty(n))&&(n_in(i)~=n(i))
                    error('mode size (%d,%d) of %s is inconsistent', i, k, aname);
                end;
                ra_in(i+1,k) = size(a{i,k},4);
                if (ra_in(i,k)~=size(a{i,k},1))
                    error('TT rank (%d,%d) of %s is inconsistent', i, k, aname);
                end;
                if (nargin>5)&&(~isempty(ra))&&(ra_in(i+1,k)~=ra(i+1,k))
                    error('TT rank (%d,%d) of %s is inconsistent', i, k, aname);
                end;
            end;
        end;
    end;
    n = n_in;
    ra = ra_in;
end;
end
