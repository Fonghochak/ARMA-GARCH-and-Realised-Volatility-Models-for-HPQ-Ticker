clearvars; clc;

%Load the dataset of RV measures
load('RVtbl.mat');

%Get datetime array
dt = datetime(num2str(RVtbl{:,1}),'InputFormat','yyyyMMdd','Format','dd/MM/yyyy');

%Get the names of all series
RVnames=RVtbl.Properties.VariableNames(2:10);

%We choose the benchmark model to be RK as the l.h.s. of a HAR model
RK=RVtbl.RK;

%We then  do rolling-window forecasts using different r.h.s. variables.
%We store all the variables in a matrix below
RVmat=RVtbl{:,2:10};

%Plot the RV measures
figure('units','normalized','outerposition',[0 0 1 1]);
plot(dt, RVtbl{:,2:10});
legend(RVnames);

%%

%Choose the length of the insample period
insmpl=504;

%Initialize a matrix to collect the forecasts
res=zeros(length(RK)-insmpl, size(RVmat,2)* 2);

%Write a loop to compute the forecasts
for i=1:size(RVmat,2)
   res(:,((2*i)-1):(2*i))=HAR_frcst(RK, RVmat(:,i), insmpl);
end
%Note that for the matrix res, all the odd columns store the true value RK,
%and all the even colums store the forecasts from different models,
%in the same order as stored in the RVtbl.mat

%%

%Now to evaluate our forecasting performance for each estimator, we can
%construct MSPE, MAPE and QLIKE measures on the big res matrix:
HAR_result=HAR_eval(res);

%Store the results in a nice table
HARtable=array2table(HAR_result, 'RowNames',RVnames,'VariableNames',...
    {'MAPE','MSPE','QLIKE'});
disp(HARtable);

%From the table we see that different loss functions lead to differnt
%conclusions for the best model, so according to MAPE and MSPE, RK performs the
%best, but QLIKE suggests that preRV performs the best.
%Also, all three loss functions suggest that GARCH measure performs the worst,
%which is not surprising because GARCH measure do not take intraday price
%information into account.

%%

%Now, we can use a modified DM test to test if one model significantly
%outperforms another.
%We will be comparing HAR-RK against all possible alternatives in the dataset.
%To perform the modified DM test, we firstly compute the forecasting errors:
evec=res(:,2:2:end)-res(:,1:2:end);

%evec contains the forecasting error computed by forecast-true for each series.
%Note that the 5th series is the RK benchmark.
%To make our programme simpler, we rearrange the order of the columns and 
%move RK to the first column of evec:
evec=evec(:,[5 1:4 6:9]);

%Initialize a matrix to store all results
DMresult=zeros(8,2);
for i=1:8
    %We are testing HAR-RK against all other alternatives. Therefore the
    %first entries will always be the loss from HAR-RK
 [DM,p_value] = dmtest_modified(evec(:,1), evec(:,i+1));
 DMresult(i,:)=[DM p_value];
end

DMtbl=array2table(DMresult,'RowNames',{'RK vs tickRV'...
    'RK vs RV' 'RK vs SubRV' 'RK vs TSRV' 'RK vs preRV' 'RK vs RBV'...
    'RK vs preRBV' 'RK vs GARCH'},'VariableNames',{'tStat','p_value'});
disp(DMtbl);
%From the table we can see that all TestStats are negative, which suggests
%that the HAR-RK model outperform HAR models with other r.h.s. variables.
%However this difference in terms of MSPE is not significantly different
%from zero since the p-values are all very high, except from HAR-GARCH as
%we see a clear rejection.
%This indicates that the HAR-GARCH model performs significantly worse than
%HAR-RK in predicting RK with a much larger forecasting error.
