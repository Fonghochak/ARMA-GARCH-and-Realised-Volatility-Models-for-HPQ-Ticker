function [ret,XF]= HARX( Y,X )
%This function is used to compute the l.h.s. and r.h.s. variables for a HAR
%model: Y(t)=w+b_d*X_d(t-1)+b_w*X_w(t-1)+b_m*X_m(t-1)+u(t)
%Input: Y: a T-by-1 vector for the l.h.s. of HAR model
%       X: a T-by-1 vector for the r.h.s. of HAR model
%Output: ret: a (T-22)-by-4 matrix. [Y, X_d, X_w, X_m]
%        XF:  a 1-by-3 vector that can be used to forecast Y(t+1)

%Input Checking
if length(Y)~=length(X)
   error('Size mismatch between Y and X');
end

%Number of rows in Y
r=length(Y);

%Initialize the return matrix for forecasts
ret=zeros(r-22,4);

%Retrieve the vector of Y as l.h.s. of HAR
ret(:,1)=Y(23:r);

%Retrieve the lagged vector of X as X_d
ret(:,2)=X(22:r-1);

%Computer weekly and monthly moving averages
Xw=movmean(X,[4,0]);
Xm=movmean(X,[21,0]);

%Store the corresponding X_w and X_m
ret(:,3)=Xw(22:r-1);
ret(:,4)=Xm(22:r-1);

%Store X_d(T) X_w(T) and X_m(T) as XF. 
XF=[X(r) Xw(r) Xm(r)];
end

