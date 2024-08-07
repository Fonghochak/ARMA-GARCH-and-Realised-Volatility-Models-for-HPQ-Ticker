function [ic_tab] = ic_est(logL, npar, T)
aic=-2*logL/T+2*(npar)/T;
bic=-2*logL/T+(npar)*log(T)/T;
hqic=-2*logL/T+2*(npar)*log(log(T))/T;
% Create an array with the information criteria
ic=[aic, bic, hqic];
% Create a table with the optimal parameters
ic_tab=array2table(ic, 'VariableNames', {'AIC', 'BIC', 'HQIC'});
end