% Check for consistency and retrieve the sizes of a vector in various TT formats
%   function [d,n,Rx,rx,vectype]=grumble_vector(x,xname,d,n,Rx,rx)
%
% !!! Note !!!!
% This is a technical routine, please use higher-level amen_solve or tamen 
% unless you understand precisely what are you doing.

function [d,n,Rx,rx,vectype]=grumble_vector(x,xname,d,n,Rx,rx)
if (isempty(x))
    d = 0;
    n = 0;
    Rx = 0;
    rx = 0;
    vectype = 0;
    return;
end;
if (isa(x, 'tt_tensor'))
    if (nargin>2)&&(~isempty(d))&&(x.d~=d)
        error('dim of %s is inconsistent',xname);
    else
        d = x.d;
    end;
    if (nargin>3)&&(~isempty(n))&&((numel(x.n)~=numel(n))||(~all(x.n==n)))
        error('mode sizes of %s are inconsistent', xname);
    else
        n = x.n;
    end;
    if (nargin>4)&&(~isempty(Rx))&&(Rx~=1)
        error('canonical rank of %s is inconsistent', xname);
    else
        Rx=1;
    end;
    if (nargin>5)&&(~isempty(rx))&&((numel(x.r)~=numel(rx))||(~all(x.r==rx)))
        error('TT ranks of %s are inconsistent', xname);
    else
        rx = x.r;
    end;
    vectype = 'tt_tensor';
else
    % {d,R} format with 4 dimensions, or maybe a cell of tt_tensors
    if (nargin>4)&&(~isempty(Rx))&&(Rx~=size(x,2))
        error('canonical rank of %s is inconsistent', xname);
    else
        Rx = size(x,2);
    end;
    if (strcmp(xname, 'x')||strcmp(xname, 'z'))&&(Rx>1)
        fprintf('Extracting the last term of input %s...\n', xname);
        x = x(:,end);
        if (isa(x{1}, 'tt_tensor'))
            x = x{1};
        end;
        if (nargin<3)
            d = [];
        end;
        if (nargin<4)
            n = [];
        end;
        if (nargin<5)
            Rx = [];
        end;
        if (nargin<6)
            rx = [];
        else
            rx = rx(:,end);
        end;
        % Parse the last term
        [d,n,Rx,rx,vectype]=grumble_vector(x,xname,d,n,Rx,rx);
        return;
    end;
    if (nargin>2)&&(~isempty(d))&&(size(x,1)~=d)
        error('dim of %s is inconsistent',xname);
    else
        d = size(x,1);
    end;    
    if (nargin<=3)||(isempty(n))
        n_in = ones(d,1);
    else
        if (numel(n)~=d)
            error('mode sizes of %s are inconsistent', xname);
        end;
    end;
    if (nargin<=5)||(isempty(rx))
        rx_in = ones(d+1,Rx);
    else
        if (size(rx,1)~=(d+1))||(size(rx,2)~=Rx)
            error('TT ranks of %s are inconsistent', xname);
        end;
    end;
    for k=1:Rx
        for i=1:d
            n_in(i) = size(x{i,k},2)*size(x{i,k},3);
            if (nargin>3)&&(~isempty(n))&&(n_in(i)~=n(i))
                error('mode size (%d,%d) of %s is inconsistent', i, k, xname);
            end;
            rx_in(i+1,k) = size(x{i,k},4);
            if (rx_in(i,k)~=size(x{i,k},1))
                error('TT rank (%d,%d) of %s is inconsistent', i, k, xname);
            end;
            if (nargin>5)&&(~isempty(rx))&&(rx_in(i+1,k)~=rx(i+1,k))
                error('TT rank (%d,%d) of %s is inconsistent', i, k, xname);
            end;
        end;
    end;
    n = n_in;
    rx = rx_in;
    vectype = 'dR';
end;
end
