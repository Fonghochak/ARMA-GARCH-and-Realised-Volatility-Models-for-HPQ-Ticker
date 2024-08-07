function ret= HAR_frcst( Y,X,insmpl )
%This function computes one-step ahead rolling window HAR forecasts with
%re-estimation with the HAR model:
%Y(t)=w+b_d*X_d(t-1)+b_w*X_w(t-1)+b_m*X_m(t-1)+u(t)
%Inputs: Y: a T-by-1 vector used as the l.h.s. of the HAR model
%        X: a T-by-1 vector used as the r.h.s. of the HAR model
%        insmpl: a integer indicating the size of the initial estimation
%        window
%Outputs: ret: a (T-insmpl)-by-2 vector. The first column is the true value
%and the second column stores the forecasted value

%Input Checking
if length(Y)~=length(X)
   error('Size mismatch between Y and X');
end
if insmpl>=length(Y)
   error('In-sample period should be shorter than the length of the data'); 
end

%Extract the length of the dataset
T=length(Y);

%Get the number of re-estimations of the HAR model
N=T-insmpl;

%Initialize a vector to compute results
ret=zeros(N,2);

%Set the first column of ret to be true values
ret(:,1)=Y(insmpl+1:T);

for i=1:N
    %Get temporary Y and X for each re-estimation of HAR. Think about the
    %index i:insmpl+i-1!
    Ytemp=Y(i:insmpl+i-1);
    Xtemp=X(i:insmpl+i-1);
    
    %Compute the r.h.s. and l.h.s. variables using the HARX.m function
    [h,XF]= HARX(Ytemp, Xtemp);
    
    %OLS regression by hand...
     har=[ ones(insmpl-22,1) h(:,2:4)]; %=X
     b=(har'*har)\har'*h(:,1); %b=(X'X)^(-1)*X'*y
     
     %Compute the one-step ahead forecast and store it
     ret(i,2)=[1 XF]*b;
end
end

