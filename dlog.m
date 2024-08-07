%This function computes the log difference between the i-th row and the
%(i-1)-th row of a matrix for i=2:end
%The first element is padded with zeroes

function ret=dlog(X)
c=size(X,2);
ret=[zeros(1,c);diff(log(X))];
end