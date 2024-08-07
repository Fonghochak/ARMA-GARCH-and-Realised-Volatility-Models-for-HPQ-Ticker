function out=ADF_auto(data,dif,max_lags,sel_method)
%This function computes the augmented Dickey Fuller (ADF test with automatic lag
%selection based on an information criteria from AIC, BIC and HQIC. Three
%sets of tests are performed: the ADF test with no drift or trend, with a
%drift and with both drift and trend.
%Inputs:
%data: T-by-1 vector of data
%dif: non-negative integer, the ADF test is performed after differencing
%the original series 'dif' times.
%max_lags: non-negative integer, the maximum lags included in the ADF test
%selection_method: choose from 'AIC', 'BIC' and 'HQIC', the information
%criteria minimized to select an optimal lag length
%output: A 6-by-3 table documenting the test statistics, critical values, pvalues and 
%the optimal lags chosen
%Checking selection methods
sel_method_choice={'AIC' 'BIC' 'HQIC'};
if ismember(sel_method,sel_method_choice)==0
   error('Selection method not supported.');
end
%Checking input type of lags (must be integers)
if rem(dif,1)~=0 || rem(max_lags,1)~=0
error('Wrong input type.');
end
%Differencing the data
sdata=data;
while dif>0
sdata=diff(data,dif);
dif=dif-1;
end
warning('off');
%Do an ADF test for 0:max_lags with no drift or trend
[~,~,~,~,regar]=adftest(sdata,'Model','AR', 'lags', 0:max_lags);
%Do an ADF test for 0:max_lags with drift but no trend
[~,~,~,~,regard]=adftest(sdata,'Model','ARD', 'lags', 0:max_lags);
%Do an ADF test for 0:max_lags with drift and trend
[~,~,~,~,regts]=adftest(sdata,'Model','TS', 'lags', 0:max_lags);
%Store the information criteria in three different matrices
ICAR=[(regar.AIC); (regar.BIC); (regar.HQC)];
ICARD=[(regard.AIC); (regard.BIC); (regard.HQC)];
ICTS=[(regts.AIC); (regts.BIC); (regts.HQC)];
%Choosing the minimum information criteria from the corresponding matrix
[~, minICARi]=min(ICAR,[],2);
[~, minICARDi]=min(ICARD,[],2);
[~,minICTSi]=min(ICTS,[],2);
%Initialize the matrix Qtable to store results
Qtable=zeros(3,6);
%Choose the corresponding best lag according to the selcted information
%criterion
if strcmp(sel_method,'AIC')==1
lagar=minICARi(1);
lagard=minICARDi(1);
lagts=minICTSi(1);
elseif strcmp(sel_method,'BIC')==1
lagar=minICARi(2);
lagard=minICARDi(2);
lagts=minICTSi(2);
else
lagar=minICARi(3);
lagard=minICARDi(3);
lagts=minICTSi(3);
end
% Use the chosen lag to produce ADF test results
[~,p1,df1,cv1,~] = adftest(sdata,'model','AR','alpha',[0.1 0.05 0.01],'lags',lagar);
[~,p2,df2,cv2,~] = adftest(sdata,'model','ARD','alpha',[0.1 0.05 0.01],'lags',lagard);
[~,p3,df3,cv3,~] = adftest(sdata,'model','TS','alpha',[0.1 0.05 0.01],'lags',lagts);
%Store the relevant statistics in a table
Qtable(1,1)=df1(1);
Qtable(2,1)=df2(1);
Qtable(3,1)=df3(1);
Qtable(1,2:4)=cv1;
Qtable(2,2:4)=cv2;
Qtable(3,2:4)=cv3;
Qtable(1,5)=p1(1);
Qtable(2,5)=p2(1);
Qtable(3,5)=p3(1);
Qtable(1,6)=lagar-1;
Qtable(2,6)=lagard-1;
Qtable(3,6)=lagts-1;
%Prepare the output.
out=array2table(Qtable','RowNames', {'Test Stat' ' 10% C.V.' '5% C.V.' '1%C.V.' 'P-value' 'Lag used'},...
'VariableNames', {'None' 'Drift' 'Trend'});
end