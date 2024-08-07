function [info_table, IC, params, params_table] = ARMA_GARCH_optimal(data,pmax, qmax, gmax, amax, dist, model)
% This function computes the optimal AR and MA lags for an ARMA(p,q) model
% and the best GARCH and ARCH lags for a GARCH(g,a) model
% by minimizing information criteria.
% Inputs:
% data: T-by-1 vector of data to be modeled by an ARMA(p,q) model
% pmax: integer. Maximum AR lags considered in the model
% qmax: integer. Maximum MA lags considered in the model
% gmax: integer. Maximum GARCH lags considered in the model
% amax: integer. Maximum ARCH lags considered in the model
% dist: string. Distribution assumed.
% Choice:
% 'Gaussian': Normal/Gaussian Distribution
% 't': t-Distribution
% model: string. GARCH-type model for estimation.
% Choice:
% 'GARCH': Symmetric GARCH model
% 'EGARCH': Assymetric EGARCH model
% 'GJR-GARCH': Assymetric GJR-GARCH/TGARCH model
% Outputs:
% ICarray: array containing optimal ARMA(p,q) and GARCH(g,a) lags
% ICtable: table containing optimal ARMA(p,q) and GARCH(g,a) lags

% Initiallize 1st stage information criteria matrices
AICmat1=zeros(1+gmax,amax);
BICmat1=zeros(1+gmax,amax);
HQICmat1=zeros(1+gmax,amax);

% Initiallize 2nd stage information criteria matrices
AICmat2=zeros(pmax+1,qmax+1);
BICmat2=zeros(pmax+1,qmax+1);
HQICmat2=zeros(pmax+1,qmax+1);

% Initiallize matrices for GARCH(m,n) parameters
gAICmat=zeros(pmax+1,qmax+1);
aAICmat=zeros(pmax+1,qmax+1);
gBICmat=zeros(pmax+1,qmax+1);
aBICmat=zeros(pmax+1,qmax+1);
gHQICmat=zeros(pmax+1,qmax+1);
aHQICmat=zeros(pmax+1,qmax+1);

%Restrict the possible number of parameters:
T=length(data);
if amax+1+gmax>=T/10
error('Too many parameters. Reduce pmax and qmax');
end

% Define a function to select the type of volatility model
function [vol, lev] = func_garch(model, g, a)
if strcmp(model, 'GARCH') || isempty(model)
vol=garch(g,a);
lev=0;
elseif strcmp(model, 'EGARCH')
vol=egarch(g,a);
lev=a;
elseif strcmp(model, 'GJR-GARCH')
vol=gjr(g,a);
lev=a;
else
error('Invalid model type specified.');
end
end

% Compute the log-likelihood and info criteria for the models considered
% We start with ARMA(0,0) and loop over all the models up to ARMA(pmax,qmax)
for i=1:pmax+1
for j=1:qmax+1
% We loop from GARCH(0,0) to GARCH(gmax,amax) for each ARMA model
for m=1:gmax+1
for n=1:amax+1
%If it is GARCH(0,0), treat it as an ARMA model
if m*n==1
Mdltemp=arima(i-1,0,j-1);
Mdltemp.Distribution = dist;
[~,~,logL,~]=estimate(Mdltemp,data,'Display','off');
lev=0;
elseif m>1
if n>1
p=i-1;
q=j-1;
if i==1
p=[];
end
if j==1
q=[];
end
[vol, lev]=func_garch(model,m-1,n-1);
Mdltemp=arima('ARLags',p,'MALags',q,'Variance',vol);
Mdltemp.Distribution = dist;
[~,~,logL,~]=estimate(Mdltemp,data,'Display','off');
else
% If it is a GARCH(i,0) model, return negative infinity as
% likelihood
logL=-Inf;
lev=0;
end
end


% Write the information criteria onto matrices
% to later select the optimal GARCH parameters
AICmat1(m,n)=-2*logL/T+2*(i+j+m+n-4+lev)/T;
BICmat1(m,n)=-2*logL/T+(i+j+m+n-4+lev)*log(T)/T;
HQICmat1(m,n)=-2*logL/T+2*(i+j+m+n-4+lev)*log(log(T))/T;
end
end
% Write the information criteria for the optimal GARCH model
% onto matrices to later select the optimal ARMA parameters
AICmat2(i,j)=min(min(AICmat1));
BICmat2(i,j)=min(min(BICmat1));
HQICmat2(i,j)=min(min(HQICmat1));
% Find the best GARCH parameters for each ARMA model
[rowAIC1,colAIC1]=find(AICmat1==min(min(AICmat1)));
[rowBIC1,colBIC1]=find(BICmat1==min(min(BICmat1)));
[rowHQIC1,colHQIC1]=find(HQICmat1==min(min(HQICmat1)));
% Write the best GARCH parameters for each ARMA model
% onto matrices
gAICmat(i,j)=rowAIC1-1;
aAICmat(i,j)=colAIC1-1;
gBICmat(i,j)=rowBIC1-1;
aBICmat(i,j)=colBIC1-1;
gHQICmat(i,j)=rowHQIC1-1;
aHQICmat(i,j)=colHQIC1-1;
end
end
% Find the minimum information criteria
minAIC=min(min(AICmat2));
minBIC=min(min(BICmat2));
minHQIC=min(min(HQICmat2));
% Find the location of the minimum information criteria
[rowAIC2,colAIC2]=find(AICmat2==minAIC);
[rowBIC2,colBIC2]=find(BICmat2==minBIC);
[rowHQIC2,colHQIC2]=find(HQICmat2==minHQIC);
% Find the optimal ARMA parameters
pAIC=rowAIC2-1;
qAIC=colAIC2-1;
pBIC=rowBIC2-1;
qBIC=colBIC2-1;
pHQIC=rowHQIC2-1;
qHQIC=colHQIC2-1;
% Find the optimal GARCH parameters
gAIC=gAICmat(rowAIC2,colAIC2);
aAIC=aAICmat(rowAIC2,colAIC2);
gBIC=gBICmat(rowBIC2,colBIC2);
aBIC=aBICmat(rowBIC2,colBIC2);
gHQIC=gHQICmat(rowHQIC2,colHQIC2);
aHQIC=aHQICmat(rowHQIC2,colHQIC2);

% Create a table with information about the model specification
info_table=array2table({model; dist}, 'VariableNames', {'Info'}, 'RowNames',{'Model', 'Distribution'});
% Create an array with the information criteria
IC=[minAIC, minBIC, minHQIC];
% Create an array with the optimal parameters
params=[pAIC, qAIC, gAIC, aAIC, minAIC; pBIC, qBIC, gBIC, aBIC, minBIC;
pHQIC, qHQIC, gHQIC, aHQIC, minHQIC];
% Create a table with the optimal parameters
params_table=array2table(params, 'VariableNames', {'AR(p)', 'MA(q)','GARCH(g)', 'ARCH(a)', 'IC'}, 'RowNames', {'AIC', 'BIC', 'HQIC'});
end