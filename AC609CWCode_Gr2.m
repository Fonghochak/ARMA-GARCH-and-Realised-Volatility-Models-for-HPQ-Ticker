%%                         PART 1: ARMA-GARCH Model                     %%
       %%  a.1) Tick-by-tick and open-to-close series Calculation %%
% load an .csv file into MATLAB as table
HPdata = readtable('HPQ_trade_quote.csv');

% extract the numerical part of the data using curly brackets to return
% a numerical matrix

HPdata_matrix = HPdata{:,:};
HPprice_matrix = HPdata{:,4};
HPdate_matrix=HPdata{:,1};

%transform timestamp of tbt series into even order with a new column 'timescale'
timescale = double(HPdata{:,1} + HPdata{:,3}/23400);
uniquedate=unique(HPdata{:,1});
T = length(uniquedate);
T2 = length(timescale);
% Compute the (1) tick-by-tick return and (2) open-to-close return
% Matlab function files: (1) dlog.m
% additionally, we squared log return of both cases
HPreturn_tbt = dlog(HPprice_matrix);
HPreturn_tbt_squared = HPreturn_tbt.^2;

HPreturn_OtC = zeros(T,1);
% Loop over each day to compute the open-to-close return
for i=1:T
    temp=HPprice_matrix(HPdate_matrix==uniquedate(i));
    HPreturn_OtC(i) = log(temp(end))-log(temp(1));
end

HPreturn_Otc_squared = HPreturn_OtC.^2;

HPreturn_tbt = round(HPreturn_tbt, 10);
HPreturn_OtC = round(HPreturn_OtC, 10);
HPreturn_tbt_squared = round(HPreturn_tbt, 10);
HPreturn_OtC_squared = round(HPreturn_OtC, 10);

                %%  b.1) Descriptive Analysis Calculations %%
names = {'HPreturn_tbt', 'HPreturn_OtC',... 
        'HPreturn_tbt_squared', 'HPreturn_Otc_squared'};

% Computing descriptive statistics
% Mean
meanHPret_tbt=mean(HPreturn_tbt);
meanHPret_OtC=mean(HPreturn_OtC);
meanHPreturn_tbt_squared=mean(HPreturn_tbt_squared);
meanHPreturn_Otc_squared=mean(HPreturn_Otc_squared);
% Variance
varHP_tbt=var(HPreturn_tbt);
varHP_OtC=var(HPreturn_OtC);
varHP_tbt_squared=var(HPreturn_tbt_squared);
varHP_otC_squared=var(HPreturn_Otc_squared);
% Maximum and miminum
maxHPret_tbt=max(HPreturn_tbt);
minHPret_tbt=min(HPreturn_tbt);
maxHPret_OtC=max(HPreturn_OtC);
minHPret_OtC=min(HPreturn_OtC);
maxHPreturn_tbt_squared=max(HPreturn_tbt_squared);
minHPreturn_tbt_squared=min(HPreturn_tbt_squared);
maxHPreturn_Otc_squared=max(HPreturn_Otc_squared);
minHPreturn_Otc_squared=min(HPreturn_Otc_squared);
% Standard deviation
std_HPreturn_tbt=std(HPreturn_tbt);
std_HPreturn_OtC=std(HPreturn_OtC);
std_HPreturn_tbt_squared=std(HPreturn_tbt_squared);
std_HPreturn_Otc_squared=std(HPreturn_Otc_squared);
% Skewness
ske_HPreturn_tbt=skewness(HPreturn_tbt);
ske_HPreturn_OtC=skewness(HPreturn_OtC);
ske_HPreturn_tbt_squared=skewness(HPreturn_tbt_squared);
ske_HPreturn_Otc_squared=skewness(HPreturn_Otc_squared);
% Kurtosis
kur_HPreturn_tbt=kurtosis(HPreturn_tbt);
kur_HPreturn_OtC=kurtosis(HPreturn_OtC);
kur_HPreturn_tbt_squared=kurtosis(HPreturn_tbt_squared);
kur_HPreturn_Otc_squared=kurtosis(HPreturn_Otc_squared);
% Quantiles
s_qvector=[.01 .05 .1 .5 .9 .95 .99];
q_HPreturn_tbt=quantile(HPreturn_tbt,s_qvector);
q_HPreturn_OtC=quantile(HPreturn_OtC,s_qvector);
q_HPreturn_tbt_squared=quantile(HPreturn_tbt_squared,s_qvector);
q_HPreturn_Otc_squared=quantile(HPreturn_Otc_squared,s_qvector);

% Prepare a table for the results
% Combine the statistics first:
stat_tick=[meanHPret_tbt; varHP_tbt; maxHPret_tbt; minHPret_tbt; std_HPreturn_tbt; ske_HPreturn_tbt;
kur_HPreturn_tbt; q_HPreturn_tbt'];
stat_daily=[meanHPret_OtC; varHP_OtC; maxHPret_OtC; minHPret_OtC; std_HPreturn_OtC;
ske_HPreturn_OtC; kur_HPreturn_OtC; q_HPreturn_OtC'];
stat_tick_sq=[meanHPreturn_tbt_squared; varHP_tbt_squared; maxHPreturn_tbt_squared; minHPreturn_tbt_squared;
std_HPreturn_tbt_squared; ske_HPreturn_tbt_squared; kur_HPreturn_tbt_squared; q_HPreturn_tbt_squared'];
stat_otc_sq=[meanHPreturn_Otc_squared; varHP_otC_squared; maxHPreturn_Otc_squared; minHPreturn_Otc_squared; std_HPreturn_Otc_squared;
ske_HPreturn_Otc_squared; kur_HPreturn_Otc_squared; q_HPreturn_Otc_squared'];
s_stat=[stat_tick, stat_daily, stat_tick_sq, stat_otc_sq];
s_stattable=array2table(s_stat, 'VariableNames', names, 'RowNames', {'Mean'...
'Variance' 'Maximum' 'Minimum' 'Std.Dev.' 'Skewness' 'Kurtosis' ...
'Q(0.01)' 'Q(0.05)' 'Q(0.1)' 'Q(0.5)' 'Q(0.9)' 'Q(0.95)' 'Q(0.99)'});

% Jarque-Bera test for normality
JB_HPreturn_tbt=(length(HPreturn_tbt))/6*(ske_HPreturn_tbt.^2+0.25*(kur_HPreturn_tbt-3).^2);
JB_HPreturn_OtC=(length(HPreturn_OtC))/6*(ske_HPreturn_OtC.^2+0.25*(kur_HPreturn_OtC-3).^2);
JB=[JB_HPreturn_tbt, JB_HPreturn_OtC];
JBpvalue=chi2cdf(JB,2,'upper');

               %%  b.2) Plots for Descriptive Analysis %%

% 5 stylised facts (page 9 Script)
    % (1): Non-normality of returns & Log-non-normality of prices
    % (2): Heavy tails
    % (3): Volatility clusters
    % (4): Predictability of volatility over returns
    % (5): Typical trading-session effects of intraday returns
               

             %% b.2.1: Non-normality of returns series %%

% Create a figure structure
fig2=figure;
% Set size of the plots [left bottom width height]
plotsize2=[200,200,1600,1200];
fig2.Position=plotsize2;

% 1. Subplot for tick-by-tick log returns
subplot(2,2,1);
% Histogram
histogram(HPreturn_tbt);
% Set x-axis limits
xlim([-0.0003 0.0003]);
% Add title
title('HP Tick-by-tick Log Returns');
% Add x-axis label
xlabel('Log Returns');
% Add y-axis label
ylabel('Count');

% 2. Subplot for open-to-close log returns
subplot(2,2,2);
% Histogram
histfit(HPreturn_OtC);
% Set x-axis limits
xlim([-0.08 0.08]);
% Add title
title('HP Open-to-close Log Returns');
% Add x-axis label
xlabel('Log Returns');
% Add y-axis label

                        %% b.2.2: Heavy tails %%

%3 Q-Q plot for Returns tick-by-tick
subplot(2,2,3);
qqplot(HPreturn_tbt);
title('QQPlot Tick-by-tick Returns');
%4 Q-Q plot for Returns Open-to-close
subplot(2,2,4);
qqplot(HPreturn_OtC);
title('QQPlot Open-to-close Returns');

                    %% b.2.3: Volatility clusters %%

% Create a figure structure
fig1=figure;
% Set size of the plots [left bottom width height]
plotsize=[200,200,1600,1200];
fig1.Position=plotsize;

% Subplot for tick-by-tick log returns
subplot(2,2,1);
% Line plot
plot(HPreturn_tbt);
% Set x-axis limits
xlim([0 T2]);
% Add title
title('HP Tick-by-tick Log Returns');
% Add x-axis label
xlabel('Intraday (ticks)');
% Add y-axis label
ylabel('Tick-by-tick Log Returns');

% Subplot for open-to-close log returns
subplot(2,2,2);
% Line plot
plot(uniquedate, HPreturn_OtC);
% Set x-axis limits
xlim([0 T]);
% Add title
title('HP Open-to-close Log Returns');
% Add x-axis label
xlabel('Days');
% Add y-axis label
ylabel('Open-to-close Log Returns');

% Subplot for tick-by-tick squared log returns
subplot(2,2,3);
% Line plot
plot(HPreturn_tbt_squared);
% Set x-axis limits
xlim([0 T2]);
% Add title
title('HP Tick-by-tick Squared Log Returns');
% Add x-axis label
xlabel('Intraday (ticks)');
% Add y-axis label
ylabel('Squared Log Returns');

% Subplot for open-to-close squared log returns
subplot(2,2,4);
% Line plot
plot(uniquedate, HPreturn_Otc_squared);
% Set x-axis limits
xlim([0 T]);
% Add title
title('HP Open-to-close Squared Log Returns');
% Add x-axis label
xlabel('Days');
% Add y-axis label
ylabel('Squared Log Returns');

      %% b.2.4: Superior predictability of volatility over returns %%

% Create a figure structure
fig3=figure;
% Set size of the plots [left bottom width height] 
plotsize3=[200,200,1600,1200];
fig3.Position=plotsize3;

% Set up a matrix of zeros for later
Q=zeros(Lags,4);
% Creating the plot for tick-by-tick log returns
% Computing autocorrelation and partial autocorrelation
[acf,lag1,bound1]=autocorr(HPreturn_tbt,'NumLags',Lags,'NumSTD',1.96);
[pacf,lag2,bound2]=parcorr(HPreturn_tbt,'NumLags',Lags,'NumSTD',1.96);
% Compute Ljung-Box test statistics for later
Q(:,1)=length(HPreturn_tbt)*(length(HPreturn_tbt)+2)*cumsum((acf(2:Lags+1).^2)./(length(HPreturn_tbt)-lag1(2:Lags+1)));
% Subplot for tick-by-tick log returns
subplot(2,2,1);
hold on; % Adding new plots to the existing plot
p1=stem(1:1:Lags,acf(2:Lags+1),'filled'); % Plotting acf
p2=stem(1:1:Lags,pacf(2:Lags+1)); % Plotting pacf
p=plot(1:1:Lags,[ones(Lags,1).*bound1(1) ones(Lags,1).*bound1(2)]); % Plotting confidence bounds
hold off;
% Graph settings
p1.Color='r';
p1.MarkerSize=4;
p2.Color='b';
p2.MarkerSize=4;
p(1).LineWidth=1;
p(1).LineStyle='--';
p(1).Color='k';
p(2).LineWidth=1;
p(2).LineStyle='--';
p(2).Color='k';
% Add title
title('HP Tick-by-tick Log Returns');
% Add x-axis label
xlabel('Lags');
% Add y-axis label
ylabel('Autocorrelation');

% Creating the plot for open-to-close log returns
% Computing autocorrelation and partial autocorrelation
[acf,lag3,bound3]=autocorr(HPreturn_OtC,'NumLags',Lags,'NumSTD',1.96);
[pacf,lag4,bound4]=parcorr(HPreturn_OtC,'NumLags',Lags,'NumSTD',1.96);
% Compute Ljung-Box test statistics for later
Q(:,2)=length(HPreturn_OtC)*(length(HPreturn_OtC)+2)*cumsum((acf(2:Lags+1).^2)./(length(HPreturn_OtC)-lag3(2:Lags+1)));
% Subplot for open-to-close log returns
subplot(2,2,2);
hold on; % Adding new plots to the existing plot
p1=stem(1:1:Lags,acf(2:Lags+1),'filled'); % Plotting acf
p2=stem(1:1:Lags,pacf(2:Lags+1)); % Plotting pacf
p=plot(1:1:Lags,[ones(Lags,1).*bound3(1) ones(Lags,1).*bound3(2)]); % Plotting confidence bounds
hold off;
% Graph settings
p1.Color='r';
p1.MarkerSize=4;
p2.Color='b';
p2.MarkerSize=4;
p(1).LineWidth=1;
p(1).LineStyle='--';
p(1).Color='k';
p(2).LineWidth=1;
p(2).LineStyle='--';
p(2).Color='k';
% Add title
title('HP Open-to-close Log Returns');
% Add x-axis label
xlabel('Lags');
% Add y-axis label
ylabel('Autocorrelation');
% Creating the plot for tick-by-tick squared log returns

% Computing autocorrelation and partial autocorrelation
[acf,lag5,bound5]=autocorr(HPreturn_tbt_squared,'NumLags',Lags,'NumSTD',1.96);
[pacf,lag6,bound6]=parcorr(HPreturn_tbt_squared,'NumLags',Lags,'NumSTD',1.96);
% Compute Ljung-Box test statistics for later
Q(:,3)=length(HPreturn_tbt_squared)*(length(HPreturn_tbt_squared)+2)*cumsum((acf(2:Lags+1).^2)./(length(HPreturn_tbt_squared)-lag5(2:Lags+1)));
% Subplot for daily log returns
subplot(2,2,3);
hold on; % Adding new plots to the existing plot
p1=stem(1:1:Lags,acf(2:Lags+1),'filled'); % Plotting acf
p2=stem(1:1:Lags,pacf(2:Lags+1)); % Plotting pacf
p=plot(1:1:Lags,[ones(Lags,1).*bound5(1) ones(Lags,1).*bound5(2)]); % Plotting confidence bounds
hold off;
% Graph settings
p1.Color='r';
p1.MarkerSize=4;
p2.Color='b';
p2.MarkerSize=4;
p(1).LineWidth=1;
p(1).LineStyle='--';
p(1).Color='k';
p(2).LineWidth=1;
p(2).LineStyle='--';
p(2).Color='k';
% Add title
title('HP Tick-by-tick Squared Log Returns');
% Add x-axis label
xlabel('Lags');
% Add y-axis label
ylabel('Autocorrelation');
% Creating the plot for open-to-close squared log returns

% Computing autocorrelation and partial autocorrelation
[acf,lag7,bound7]=autocorr(HPreturn_Otc_squared,'NumLags',Lags,'NumSTD',1.96);
[pacf,lag8,bound8]=parcorr(HPreturn_Otc_squared,'NumLags',Lags,'NumSTD',1.96);
% Compute Ljung-Box test statistics for later
Q(:,4)=length(HPreturn_Otc_squared)*(length(HPreturn_Otc_squared)+2)*cumsum((acf(2:Lags+1).^2)./(length(HPreturn_Otc_squared)-lag7(2:Lags+1)));
% Subplot for daily log returns
subplot(2,2,4);
hold on; % Adding new plots to the existing plot
p1=stem(1:1:Lags,acf(2:Lags+1),'filled'); % Plotting acf
p2=stem(1:1:Lags,pacf(2:Lags+1)); % Plotting pacf
p=plot(1:1:Lags,[ones(Lags,1).*bound7(1) ones(Lags,1).*bound7(2)]); % Plotting confidence bounds
hold off;
% Graph settings
p1.Color='r';
p1.MarkerSize=4;
p2.Color='b';
p2.MarkerSize=4;
p(1).LineWidth=1;
p(1).LineStyle='--';
p(1).Color='k';
p(2).LineWidth=1;
p(2).LineStyle='--';
p(2).Color='k';
% Add title
title('HP Open-to-close Squared Log Returns');
% Add x-axis label
xlabel('Lags');
% Add y-axis label
ylabel('Autocorrelation');
% Legend settings
legend('ACF', 'PACF','99% Confidence Bounds','Location','southoutside','Orientation','horizontal');


    %% b.2.5: Market microstructure effects of intraday returns %%

% 2 trading days: 7318
% 1 trading days: 3450
% first 100 transactions

% Create a figure structure
fig = figure;

% Set size of the plots [left bottom width height]
plotsize = [200, 200, 1600, 1200];
fig.Position = plotsize;

% Subplot for prices for 2 trading days
subplot(3, 1, 1);
% Line plot
plot(HPprice_matrix(1:7318));  % Use parentheses for indexing
% Set x-axis limits
xlim([0 7318]);
% Add title
title('HPQ tick-by-tick series');
% Add x-axis label
xlabel('Ticks since 9:30 on 03/01/20 (2 trading days)');
% Add y-axis label
ylabel('Prices in USD');

% Subplot for prices of the first 100 transactions
subplot(3, 1, 2);
% Line plot
plot(HPprice_matrix(1:100));  % Use parentheses for indexing
% Set x-axis limits
xlim([0 100]);
% Add x-axis label
xlabel('Ticks since 9:30 (first 100 transactions)');
% Add y-axis label
ylabel('Prices in USD');

% Subplot for tick-by-tick squared return
subplot(3, 1, 3);
% Line plot
plot(HPreturn_tbt_squared(1:3450));  % Use parentheses for indexing
% Set x-axis limits
xlim([0 3450]);
% Add x-axis label
xlabel('Ticks since 9:30 (during the day)');
% Add y-axis label
ylabel('Squared Log Returns');
% Set y-axis minor ticks
yticks(min(HPreturn_tbt_squared(1:3450)):0.1:max(HPreturn_tbt_squared(1:3450)));  % Adjust the step size as needed
grid on;  % Enable grid to show minor ticks


       %%  c.1) ARMA-GARCH Specification of Open-to-close series %%

             %%  c.1.1) Box-Jenkins Analysis for ARMA-GARCH %%

%% Step 1: Unit root test using 3 tests

% Set the significance level of the test
alpha=[0.1,0.05,0.01];
% Augmented Dickey-Fuller test with automatic lag selection based on
% information criterion => read the ADF_auto.m function for description
dfout=ADF_auto(HPreturn_OtC,0,30,'HQIC');
% Lag selection for PP and KPSS test
Bandwidth=floor(4*(length(HPreturn_OtC)^(2/9)));
% Philip Perron test
% Perform three sets of tests for three different model specifications
[~,pValue1,stat1,cValue1,~] = pptest(HPreturn_OtC, 'Model', 'AR', 'alpha', alpha, 'lags', Bandwidth);
[~,pValue2,stat2,cValue2,~] = pptest(HPreturn_OtC, 'Model', 'ARD', 'alpha', alpha, 'lags', Bandwidth);
[~,pValue3,stat3,cValue3,~] = pptest(HPreturn_OtC, 'Model', 'TS', 'alpha', alpha, 'lags', Bandwidth);
% Prepare a table to present the test outputs
pptable=[stat1(1) cValue1 pValue1(1); stat2(1) cValue2 pValue2(1) ;stat3(1)...
cValue1 pValue3(1)];
ppout = array2table(pptable', 'RowNames', {'Test Stat', '10% C.V.', '5% C.V.', '1% C.V.', 'P-value'}, ...
'VariableNames', {'None', 'Drift', 'Trend'});
% KPSS test
[~,pValue4,stat4,cValue4,~]=kpsstest(HPreturn_OtC, 'alpha', alpha,'lags', ...
Bandwidth, 'trend', false);
[~,pValue5,stat5,cValue5,~]=kpsstest(HPreturn_OtC, 'alpha', alpha,'lags', ...
Bandwidth, 'trend', true);
% Prepare a table to present the test outputs
kpsstable=[stat4(1) cValue4 pValue4(1); stat5(1) cValue5 pValue5(1)];
kpssout=array2table(kpsstable','RowNames', {'Test Stat' ' 10% C.V.' '5% C.V.'...
'1% C.V.' 'P-value' },...
'VariableNames', {'Level' 'Trend'});

% Print the results in the command window
fprintf('ADF test with optimal lag selection using %s\n');
disp(dfout);
disp('PP test with Newey-West fixed Bandwidth');
disp(ppout);
disp('KPSS test with Newey-West fixed Bandwidth');
disp(kpssout);

%% Step 2: ARMA (p,q) Inspection

% Autocorrelogram: (1) Return and (2) Volatility proxy (squared returns)
% Set number of lags
L2 = 200;
% Create a figure structure
fig4 = figure;
% Set size of the plots [left bottom width height]
plotsize4=[200,200,1200,1600];
fig4.Position=plotsize4;

% Creating the plot for open-to-close log returns
subplot(1,2,1);
% Computing autocorrelation
autocorr(HPreturn_OtC,'NumLags',L2,'NumSTD',1.96);
% Add title
title('Autocorrelation of HP daily open to close log returns');
% Add x-axis label
xlabel('Lags');
% Add y-axis label
ylabel('Autocorrelation');

% Creating the plot for open-to-close squared log returns
subplot(1,2,2);
% Computing autocorrelation
autocorr(HPreturn_OtC.^2,'NumLags',L2,'NumSTD',1.96);
% Add title
title('Autocorrelation of HP daily open to close Squared Log Returns');
% Add x-axis label
xlabel('Lags');
% Add y-axis label
ylabel('Autocorrelation');
% Legend settings
legend('ACF', '99% Confidence Bounds','Location','southoutside','Orientation','horizontal');

%% Step 3: ARMA(p,q) Parameter Optimisation
% Set the max AR(p) lag
pmax=21; 
% Set the max MA(q) lag
qmax=21;
% compute the log-likelihood and information criteria for the models considered

% use the function developed by Dr. Kevin Sheppard in his MFEToolbox
% start with ARMA(1,1) and loop over all the models to
[pbest,qbest, ICmatARMA] = ARMA_optimal(HPreturn_OtC, pmax, qmax, 'HQIC')

[~, LL_ARMA00, ~, ~, ~, ~, ~, ~] = armaxfilter(HPreturn_OtC, 1, 0, 0);
IC_ARMA00=-2*LL_ARMA00/T+2*(1)*log(log(T))/T;

[~, LL_ARMA10, ~, ~, ~, ~, ~, ~] = armaxfilter(HPreturn_OtC, 1, 1, 0);
IC_ARMA10=-2*LL_ARMA10/T+2*(1+1)*log(log(T))/T; % LOWEST Information Criteria

[~, LL_ARMA01, ~, ~, ~, ~, ~, ~] = armaxfilter(HPreturn_OtC, 1, 0, 1);
IC_ARMA01=-2*LL_ARMA01/T+2*(1+1)*log(log(T))/T;

% fit the optimal ARMA(1,1) in our sample data using Dr. Kevin MFE Toolbox
pbest = 1;
qbest = 1;
% NOTE: armaxfilter(Y,CONSTANT,P,Q,X,STARTINGVALS,OPTIONS,HOLDBACK,SIGMA2)
[parameters_ARMA11, LL_ARMA11, res_ARMA11, SEregression_ARMA11, diagnostics_ARMA11, VCVrobust_ARMA11, VCV_ARMA11, likelihoods_ARMA11] = armaxfilter(HPreturn_OtC,1,1,1);

%% Step 4: Evaluation of the optimised ARMA

%Model Evaluation

%Set the significance level of the tests
alpha=[0.1,0.05,0.01];

%we test the following assumptions:
%1. Normality of the residual series.
%Now test for normality, we use the Jacque-Bera test.
[~,p1, JBstat1,critval1]=jbtest(res_ARMA11,alpha(1));
[~,~, ~,critval2]=jbtest(res_ARMA11,alpha(2));
[~,~, ~,critval3]=jbtest(res_ARMA11,alpha(3));
jbtable=[JBstat1 critval1 critval2  critval3 p1 ];
jbout=array2table(jbtable,'VariableNames',  {'JBstat','CV10', 'CV5', 'CV1','p_value' });
disp('Jacque-Bera test on the residual');
disp(jbout);

%2. Absence of autocorrelation in the residual series.
%Ljung-Box test on the residual (for up to 21 Lags) (@ Kevin Sheppard)
Lags=21;
[q_ljungboxARMA11, pval_ljungboxARMA11] = ljungbox(res_ARMA11, Lags=21);

%3. Homoscedasticity of the residual series.  (@ Kevin Sheppard)
% ARCH LM test to check for potential heteroscedasticity in the error term
[lm_ARMA11, pval_lm_ARMA11] = lmtest1(res_ARMA11,Lags);
squared_res_ARMA11 = res_ARMA11.^2; % lagged shock (volatility proxy)
[lm_ARMA11_squared, pval_lm_ARMA11_squared] = lmtest1(squared_res_ARMA11,Lags);



%Let's look at some graphs
fig = figure('units','normalized','outerposition',[0 0 1 1]);
subplot(2,2,1);
plot(res_ARMA11);
xlim([1 T]);
title('Line plot of the residual');

subplot(2,2,2);
histogram(res_ARMA11);
title('Histogram of the residual');

%quantile-quantile plot against a standardized residual. Therefore you need
%to standardize the residual first using the zscore() function
subplot(2,2,3);
qqplot(zscore(res_ARMA11));
title('Quantile-Quantile plot of the standardized residual');
subplot(2,2,4);

hold on;
lag=100;
[acf1,~,~]=autocorr(res_ARMA11,lag);
[acf2,~,bound1]=autocorr(squared_res_ARMA11,lag);
p1=stem(1:1:lag,acf1(2:lag+1),'filled'); %plotting acf
p2=stem(1:1:lag,acf2(2:lag+1)); %plotting pacf
p=plot(1:1:lag,[ones(lag,1).*bound1(1) ones(lag,1).*bound1(2)]); %plotting confidence bounds
hold off;

%Graph settings
p1.Color='r';
p1.MarkerSize=4;
p2.Color='b';
p2.MarkerSize=4;
p(1).LineWidth=2;
p(1).LineStyle=':';
p(1).Color='k';
p(2).LineWidth=2;
p(2).LineStyle=':';
p(2).Color='k';
title('Correlogram of the residual and squared residual');
legend('Residual', 'Squared residual','95% Confidence Bounds', 'Location','southoutside','Orientation','horizontal');

% COMMENT: the residual is interpreted as demeaned returns used for later
% GARCH specification, squared residual is per se volatility proxy
% good signs: stationary, normal (QQplot), autocorrelated volatility proxy
      
        %%  c.1.2) ARMA(1,1)-GARCH(a,g) Specification %%   

% As the residual/error term distribution is not NORMAL, so Student-T is preferred
pbest = 1;  %ARMA(1,1)
qbest = 1;
amax = 21;
gmax = 21;

% we integrate 'tarch' function of Dr. Kevin into the 'GARCH_optimal'
% rather than using original lab with limited 'arima' built-in func
[abest_n,gbest_n,ICmatGARCH_n] = GARCH_optimal(res_ARMA11, pbest, qbest, amax, gmax, 'HQIC', 'NORMAL'); %get Normal GARCH(1,3)
[abest_t,gbest_t,ICmatGARCH_t] = GARCH_optimal(res_ARMA11, pbest, qbest, amax, gmax, 'HQIC', 'STUDENTST'); %get Student-t GARCH(1,1)

% compute Normal ARCH(1) to compare with optimal Normal GARCH
[~, LL_ARCH1_n, ~, ~, ~, ~, ~] = tarch(res_ARMA11, 1, 0, 0, 'NORMAL', 2);
ICmatARCH_n=-2*LL_ARCH1_n/T+2*(pbest+qbest+1)*log(log(T))/T; %HQIC
% compute StudentT ARCH(1) to compare with optimal StudentT GARCH
[~, LL_ARCH1_t, ~, ~, ~, ~, ~] = tarch(res_ARMA11, 1, 0, 0, 'STUDENTST', 2);
ICmatARCH_t=-2*LL_ARCH1_t/T+2*(pbest+qbest+1)*log(log(T))/T; %HQIC

% Fitting Normal GARCH(1,3)
abest_n = 1;
gbest_n = 3;
[parametersGARCH_n13, LL_GARCH_n13, V_GARCH_n13, ~, ~, ~, diagnosticsGARCH_n13] = tarch(res_ARMA11, abest_n, 0, gbest_n, 'NORMAL');
ICmatGARCH13=-2*LL_GARCH_n13/T+2*(1+pbest+qbest+abest_n+gbest_n)*log(log(T))/T;
% Fitting StudentT GARCH(1,1)
abest_t = 1;
gbest_t = 1;
[parametersGARCH_t11, LL_GARCH_t11, V_GARCH_t11, ~, ~, ~, diagnosticsGARCH_t11] = tarch(res_ARMA11, abest_t, 0, gbest_t, 'STUDENTST');
ICmatGARCH_t11=-2*LL_GARCH_t11/T+2*(1+pbest+qbest+abest_t+gbest_t)*log(log(T))/T;
             
%% Normal ARMA(1,1)-GARCH(1,3) Diagnosis 
GARCH_n13=arima('ARLags', 1, 'MALags', 1, 'Variance', garch(3,1));
%Estimate the model
GARCH_n13_est=estimate(GARCH_n13,HPreturn_OtC);
% Get redisuals
[res_GARCH_n13,V_GARCH_n13,LL_GARCH_n13] = infer(GARCH_n13_est,HPreturn_OtC);
% Standardize residuals
z_GARCH_n13=res_GARCH_n13./sqrt(V_GARCH_n13);

%1. residual normality
[hjb_GARCHn13,pjb_GARCHn13,statjb_GARCHn13,cvjb_GARCHn13]=jbtest(z_GARCH_n13); %h=1 reject the normality of residuals

%2. Absence of autocorrelation in the residual series.(@Kevin 2007)
%Ljung-Box test on the residual (for up to 21 Lags) (@ Kevin Sheppard)
Lags=21;
[q_ljungboxGARCH_n13, pval_ljungboxGARCH_n13] = ljungbox(z_GARCH_n13, Lags);

%3. Homoscedasticity of the residual series.  (@ Kevin Sheppard)
% ARCH LM test to check for potential heteroscedasticity in the error term
[lm_GARCH_n13, pval_lm_GARCH_n13] = lmtest1(z_GARCH_n13,Lags);


%t-distributed GARCH models
%Changing our definition of Mdl slightly
GARCH_t13=arima('ARLags',1 ,'MALags' , 1, 'Variance',garch(3,1));
GARCH_t13.Distribution='t';
%Estimate the model
GARCH_t13_est=estimate(GARCH_t13,HPreturn_OtC);
% Get redisuals
[res_GARCH_t13,V_GARCH_t13,LL_GARCH_t13] = infer(GARCH_t13_est,HPreturn_OtC);
% Standardize residuals
z_GARCH_t13=res_GARCH_t13./sqrt(V_GARCH_t13);
dof_GARCH_t13=GARCH_t13_est.Distribution.DoF;
dist_GARCH_t13=makedist('tLocationScale',0,1,dof_GARCH_t13);


%1. residual normality 
[hjb_GARCH_t13,pjb_GARCH_t13,statjb_GARCH_t13,cvjb_GARCH_t13]=jbtest(z_GARCH_t13); %h=1 reject the normality of residuals

%2. Absence of autocorrelation in the residual series.(@Kevin 2007)
%Ljung-Box test on the residual (for up to 21 Lags) (@ Kevin Sheppard)
Lags=21;
[q_ljungboxGARCH_t13, pval_ljungboxGARCH_t13] = ljungbox(z_GARCH_t13, Lags);

%3. Homoscedasticity of the residual series.  (@ Kevin Sheppard)
% ARCH LM test to check for potential heteroscedasticity in the error term
[lm_GARCH_t13, pval_lm_GARCH_t13] = lmtest1(z_GARCH_t13,Lags);


% QQplot between Normal ARMA(1,1)-GARCH(1,3) and Student-T ARMA(1,1)-GARCH(1,3)
subplot(1,2,1);
qqplot(z_GARCH_n13);
subplot(1,2,2);
qqplot(z_GARCH_t13,dist_GARCH_t13);

%% Student-T ARMA(1,1)-GARCH(1,1) Diagnosis 

%t-distributed GARCH models
%Changing our definition of Mdl slightly
GARCH_t11=arima('ARLags', 1, 'MALags', 1, 'Variance',garch(1,1));
GARCH_t11.Distribution='t';
%Estimate the model
GARCH_t11_est=estimate(GARCH_t11,HPreturn_OtC);
% Get redisuals
[res_GARCH_t11,V_GARCH_t11,LL_GARCH_t11] = infer(GARCH_t11_est,HPreturn_OtC);
% Standardize residuals
z_GARCH_t11=res_GARCH_t11./sqrt(V_GARCH_t11);
dof_GARCH_t11=GARCH_t11_est.Distribution.DoF;
dist_GARCH_t11=makedist('tLocationScale',0,1,dof_GARCH_t11);

% plot to see the goodness-of-fit of Student-T ARMA(1,1)-GARCH(1,1)
qqplot(z_GARCH_t11,dist_GARCH_t11);

%1. residual normality
[hjb_GARCH_t11,pjb_GARCH_t11,statjb_GARCH_t11,cvjb_GARCH_t11]=jbtest(z_GARCH_t11); %h=1 reject the normality of residuals

%2. Absence of autocorrelation in the residual series.(@Kevin 2007)
%Ljung-Box test on the residual (for up to 21 Lags) (@ Kevin Sheppard)
Lags=21;
[q_ljungboxGARCH_t11, pval_ljungboxGARCH_t11] = ljungbox(z_GARCH_t11, Lags);

%3. Homoscedasticity of the residual series.  (@ Kevin Sheppard)
% ARCH LM test to check for potential heteroscedasticity in the error term
[lm_GARCH_t11, pval_lm_GARCH_t11] = lmtest1(z_GARCH_t11,Lags);

% CONCLUSION: ARMA(1,1)-GARCH(1,1) with Student-t Innovations

        %%  c.1.3) Student-T ARMA(1,1)-GARCH(1,1) Competing Sets %% 

% Use ARMAX-GARCH-K-SK Toolbox of Alexandros Gabrielsen (2016)
% to get the Cond Variance, Residuals, and
 %INPUTS:
 %data:     (T x 1) vector of data
 %model:    'GARCH', 'GJR', 'EGARCH', 'NARCH', 'NGARCH, 'AGARCH', 'APGARCH',
 %          'NAGARCH'
 %distr:    'GAUSSIAN', 'T', 'GED', 'CAUCHY', 'HANSEN' and 'GC'  
 %ar:        positive scalar integer representing the order of AR
 %am:        positive scalar integer representing the order of MA
 %x:         (T x N) vector of factors for the mean process
 %p:         positive scalar integer representing the order of ARCH
 %q:         positive scalar integer representing the order of GARCH
 %y:         (T x N) vector of factors for the volatility process, must be positive!


% ARMA(1,1)-GARCH(1,1)
[parameters_GARCHt11, ~, LL_GARCHt11, V_GARCHt11, res_GARCHt11, summary_GARCHt11] = garch_set(HPreturn_OtC, 'GARCH', 'T', pbest, qbest, 0, abest_t, gbest_t, 0); %(@Alexandros Gabrielsen)

% ARMA(1,1)-EGARCH(1,1) %egarch(DATA,a,O,g,ERROR_TYPE)
[parameters_EGARCHt11, LL_EGARCHt11, V_EGARCHt11, ~, ~, ~, diagnostics_EGARCHt11] = egarch_(res_ARMA11, pbest, 0, qbest, 'STUDENTST')

% ARMA(1,1)-FIGARCH(1,1) 
[parameters_FIGARCHt11, LL_FIGARCHt11, V_FIGARCHt11, ~, ~, ~, diagnostics_FIGARCHt11] = figarch_(res_ARMA11, 1, 1, 'STUDENTST') % 1 to include the ARMA

% ARMA(1,1)-AGARCH(1,1)  - Asymmetric GARCH, Engle (1990) %agarch(EPSILON,P,Q,AGARCH or NAGARCH, ERROR_TYPE)
[parameters_AGARCHt11, ~, LL_AGARCHt11, V_AGARCHt11, res_AGARCHt11, summary_AGARCHt11] = garch_set(HPreturn_OtC, 'AGARCH', 'T', pbest, qbest, 0, abest_t, gbest_t, 0); %(@Alexandros Gabrielsen)

% ARMA(1,1)-NAGARCH(1,1) - Nonlinear Asymmetric GARCH, Engle & Ng (1993)
[parameters_NAGARCHt11, ~, LL_NAGARCHt11, V_NAGARCHt11, res_NAGARCHt11, summary_NAGARCHt11] = garch_set(HPreturn_OtC, 'NAGARCH', 'T', pbest, qbest, 0, abest_t, gbest_t, 0); %(@Alexandros Gabrielsen)

% ARMA-GJRGARCH(1,1) %tarch(EPSILON,sym,asym,GARCHlag,ERROR_TYPE,TARCH_TYPE)
[parameters_GJRGARCHt11, ~, LL_GJRGARCHt11, V_GJRGARCHt11, res_GJRGARCHt11, summary_GJRGARCHt11] = garch_set(HPreturn_OtC, 'GJR', 'T', pbest, qbest, 0, abest_t, gbest_t, 0); %(@Alexandros Gabrielsen)


%% 1. QQplot for all

dist=makedist('tLocationScale',0,1,dof_GARCH_t11);

% Standardize residuals
z_GARCH_t=res_GARCHt11./sqrt(V_GARCHt11);
z_EGARCH_t=res_ARMA11./sqrt(V_EGARCHt11);
z_FIGARCH_t=res_ARMA11./sqrt(V_FIGARCHt11);
z_AGARCH_t=res_AGARCHt11./sqrt(V_AGARCHt11);
z_NAGARCH_t=res_NAGARCHt11./sqrt(V_NAGARCHt11);
z_GJRGARCH_t=res_GJRGARCHt11./sqrt(V_GJRGARCHt11);

subplot(2,3,1)
qqplot(z_GARCH_t,dist);
% Add y-axis label
ylabel('ARMA(1,1)-GARCH(1,1)');
subplot(2,3,2)
qqplot(z_EGARCH_t,dist);
ylabel('ARMA(1,1)-EGARCH(1,1)');
subplot(2,3,3)
qqplot(z_FIGARCH_t,dist);
ylabel('ARMA(1,1)-FIGARCH(1,1)');
subplot(2,3,4)
qqplot(z_AGARCH_t,dist);
ylabel('ARMA(1,1)-AGARCH(1,1)');
subplot(2,3,5)
qqplot(z_NAGARCH_t,dist);
ylabel('ARMA(1,1)-NAGARCH(1,1)');
subplot(2,3,6)
qqplot(z_GJRGARCH_t,dist);
ylabel('ARMA(1,1)-GJRGARCH(1,1)');

%% 2. ACF for all standardised residuals

% Compute ACF
[acf_GARCH,~,bound_GARCH]=autocorr(z_GARCH_t,'NumLags',Lags,'NumSTD',1.96);
[acf_EGARCH,~,bound_EGARCH]=autocorr(z_EGARCH_t,'NumLags',Lags,'NumSTD',1.96);
[acf_FIGARCH,~,bound_FIGARCH]=autocorr(z_FIGARCH_t,'NumLags',Lags,'NumSTD',1.96);
[acf_AGARCH,~,bound_AGARCH]=autocorr(z_AGARCH_t,'NumLags',Lags,'NumSTD',1.96);
[acf_NAGARCH,~,bound_NAGARCH]=autocorr(z_NAGARCH_t,'NumLags',Lags,'NumSTD',1.96);
[acf_GJRGARCH,~,bound_GJRGARCH]=autocorr(z_GJRGARCH_t,'NumLags',Lags,'NumSTD',1.96);

[acf_GARCH2,~,bound_GARCH2]=autocorr(z_GARCH_t.^2,'NumLags',Lags,'NumSTD',1.96);
[acf_EGARCH2,~,bound_EGARCH2]=autocorr(z_EGARCH_t.^2,'NumLags',Lags,'NumSTD',1.96);
[acf_FIGARCH2,~,bound_FIGARCH2]=autocorr(z_FIGARCH_t.^2,'NumLags',Lags,'NumSTD',1.96);
[acf_AGARCH2,~,bound_AGARCH2]=autocorr(z_AGARCH_t.^2,'NumLags',Lags,'NumSTD',1.96);
[acf_NAGARCH2,~,bound_NAGARCH2]=autocorr(z_NAGARCH_t.^2,'NumLags',Lags,'NumSTD',1.96);
[acf_GJRGARCH2,~,bound_GJRGARCH2]=autocorr(z_GJRGARCH_t.^2,'NumLags',Lags,'NumSTD',1.96);

subplot(2,3,1);
hold on; % Adding new plots to the existing plot
p1=stem(1:1:Lags,acf_GARCH(2:Lags+1),'filled'); % Plotting acf
p=plot(1:1:Lags,[ones(Lags,1).*bound_GARCH(1) ones(Lags,1).*bound_GARCH(2)]); % Plotting confidence bounds
hold off;
% Graph settings
p1.Color='r';
p1.MarkerSize=4;
p(1).LineWidth=1;
p(1).LineStyle='--';
p(1).Color='k';
p(2).LineWidth=1;
p(2).LineStyle='--';
p(2).Color='k';
% Add x-axis label
xlabel('Lags');
% Add y-axis label
ylabel('ARMA(1,1)-GARCH(1,1)');

subplot(2,3,2);
hold on; % Adding new plots to the existing plot
p1=stem(1:1:Lags,acf_EGARCH(2:Lags+1),'filled'); % Plotting acf
p=plot(1:1:Lags,[ones(Lags,1).*bound_EGARCH(1) ones(Lags,1).*bound_EGARCH(2)]); % Plotting confidence bounds
hold off;
% Graph settings
p1.Color='r';
p1.MarkerSize=4;
p(1).LineWidth=1;
p(1).LineStyle='--';
p(1).Color='k';
p(2).LineWidth=1;
p(2).LineStyle='--';
p(2).Color='k';
% Add x-axis label
xlabel('Lags');
% Add y-axis label
ylabel('ARMA(1,1)-EGARCH(1,1)');

subplot(2,3,3);
hold on; % Adding new plots to the existing plot
p1=stem(1:1:Lags,acf_FIGARCH(2:Lags+1),'filled'); % Plotting acf
p=plot(1:1:Lags,[ones(Lags,1).*bound_FIGARCH(1) ones(Lags,1).*bound_FIGARCH(2)]); % Plotting confidence bounds
hold off;
% Graph settings
p1.Color='r';
p1.MarkerSize=4;
p(1).LineWidth=1;
p(1).LineStyle='--';
p(1).Color='k';
p(2).LineWidth=1;
p(2).LineStyle='--';
p(2).Color='k';
% Add x-axis label
xlabel('Lags');
% Add y-axis label
ylabel('ARMA(1,1)-FiGARCH(1,1)');

subplot(2,3,4);
hold on; % Adding new plots to the existing plot
p1=stem(1:1:Lags,acf_AGARCH(2:Lags+1),'filled'); % Plotting acf
p=plot(1:1:Lags,[ones(Lags,1).*bound_AGARCH(1) ones(Lags,1).*bound_AGARCH(2)]); % Plotting confidence bounds
hold off;
% Graph settings
p1.Color='r';
p1.MarkerSize=4;
p(1).LineWidth=1;
p(1).LineStyle='--';
p(1).Color='k';
p(2).LineWidth=1;
p(2).LineStyle='--';
p(2).Color='k';
% Add x-axis label
xlabel('Lags');
% Add y-axis label
ylabel('ARMA(1,1)-AGARCH(1,1)');

subplot(2,3,5);
hold on; % Adding new plots to the existing plot
p1=stem(1:1:Lags,acf_NAGARCH(2:Lags+1),'filled'); % Plotting acf
p=plot(1:1:Lags,[ones(Lags,1).*bound_NAGARCH(1) ones(Lags,1).*bound_NAGARCH(2)]); % Plotting confidence bounds
hold off;
% Graph settings
p1.Color='r';
p1.MarkerSize=4;
p(1).LineWidth=1;
p(1).LineStyle='--';
p(1).Color='k';
p(2).LineWidth=1;
p(2).LineStyle='--';
p(2).Color='k';
% Add x-axis label
xlabel('Lags');
% Add y-axis label
ylabel('ARMA(1,1)-NAGARCH(1,1)');

subplot(2,3,6);
hold on; % Adding new plots to the existing plot
p1=stem(1:1:Lags,acf_GJRGARCH(2:Lags+1),'filled'); % Plotting acf
p=plot(1:1:Lags,[ones(Lags,1).*bound_GJRGARCH(1) ones(Lags,1).*bound_GJRGARCH(2)]); % Plotting confidence bounds
hold off;
% Graph settings
p1.Color='r';
p1.MarkerSize=4;
p(1).LineWidth=1;
p(1).LineStyle='--';
p(1).Color='k';
p(2).LineWidth=1;
p(2).LineStyle='--';
p(2).Color='k';
% Add x-axis label
xlabel('Lags');
% Add y-axis label
ylabel('ARMA(1,1)-GJRGARCH(1,1)');

%% ACF for all squared standardised residuals

subplot(2,3,1);
hold on; % Adding new plots to the existing plot
p1=stem(1:1:Lags,acf_GARCH2(2:Lags+1),'filled'); % Plotting acf
p=plot(1:1:Lags,[ones(Lags,1).*bound_GARCH2(1) ones(Lags,1).*bound_GARCH2(2)]); % Plotting confidence bounds
hold off;
% Graph settings
p1.Color='r';
p1.MarkerSize=4;
p(1).LineWidth=1;
p(1).LineStyle='--';
p(1).Color='k';
p(2).LineWidth=1;
p(2).LineStyle='--';
p(2).Color='k';
% Add x-axis label
xlabel('Lags');
% Add y-axis label
ylabel('ARMA(1,1)-GARCH(1,1) Squared Z');

subplot(2,3,2);
hold on; % Adding new plots to the existing plot
p1=stem(1:1:Lags,acf_EGARCH2(2:Lags+1),'filled'); % Plotting acf
p=plot(1:1:Lags,[ones(Lags,1).*bound_EGARCH2(1) ones(Lags,1).*bound_EGARCH2(2)]); % Plotting confidence bounds
hold off;
% Graph settings
p1.Color='r';
p1.MarkerSize=4;
p(1).LineWidth=1;
p(1).LineStyle='--';
p(1).Color='k';
p(2).LineWidth=1;
p(2).LineStyle='--';
p(2).Color='k';
% Add x-axis label
xlabel('Lags');
% Add y-axis label
ylabel('ARMA(1,1)-EGARCH(1,1) Squared Z');

subplot(2,3,3);
hold on; % Adding new plots to the existing plot
p1=stem(1:1:Lags,acf_FIGARCH2(2:Lags+1),'filled'); % Plotting acf
p=plot(1:1:Lags,[ones(Lags,1).*bound_FIGARCH2(1) ones(Lags,1).*bound_FIGARCH2(2)]); % Plotting confidence bounds
hold off;
% Graph settings
p1.Color='r';
p1.MarkerSize=4;
p(1).LineWidth=1;
p(1).LineStyle='--';
p(1).Color='k';
p(2).LineWidth=1;
p(2).LineStyle='--';
p(2).Color='k';
% Add x-axis label
xlabel('Lags');
% Add y-axis label
ylabel('ARMA(1,1)-FiGARCH(1,1) Squared Z');

subplot(2,3,4);
hold on; % Adding new plots to the existing plot
p1=stem(1:1:Lags,acf_AGARCH2(2:Lags+1),'filled'); % Plotting acf
p=plot(1:1:Lags,[ones(Lags,1).*bound_AGARCH2(1) ones(Lags,1).*bound_AGARCH2(2)]); % Plotting confidence bounds
hold off;
% Graph settings
p1.Color='r';
p1.MarkerSize=4;
p(1).LineWidth=1;
p(1).LineStyle='--';
p(1).Color='k';
p(2).LineWidth=1;
p(2).LineStyle='--';
p(2).Color='k';
% Add x-axis label
xlabel('Lags');
% Add y-axis label
ylabel('ARMA(1,1)-AGARCH(1,1) Squared Z');

subplot(2,3,5);
hold on; % Adding new plots to the existing plot
p1=stem(1:1:Lags,acf_NAGARCH2(2:Lags+1),'filled'); % Plotting acf
p=plot(1:1:Lags,[ones(Lags,1).*bound_NAGARCH2(1) ones(Lags,1).*bound_NAGARCH2(2)]); % Plotting confidence bounds
hold off;
% Graph settings
p1.Color='r';
p1.MarkerSize=4;
p(1).LineWidth=1;
p(1).LineStyle='--';
p(1).Color='k';
p(2).LineWidth=1;
p(2).LineStyle='--';
p(2).Color='k';
% Add x-axis label
xlabel('Lags');
% Add y-axis label
ylabel('ARMA(1,1)-NAGARCH(1,1) Squared Z');

subplot(2,3,6);
hold on; % Adding new plots to the existing plot
p1=stem(1:1:Lags,acf_GJRGARCH2(2:Lags+1),'filled'); % Plotting acf
p=plot(1:1:Lags,[ones(Lags,1).*bound_GJRGARCH2(1) ones(Lags,1).*bound_GJRGARCH2(2)]); % Plotting confidence bounds
hold off;
% Graph settings
p1.Color='r';
p1.MarkerSize=4;
p(1).LineWidth=1;
p(1).LineStyle='--';
p(1).Color='k';
p(2).LineWidth=1;
p(2).LineStyle='--';
p(2).Color='k';
% Add x-axis label
xlabel('Lags');
% Add y-axis label
ylabel('ARMA(1,1)-GJRGARCH(1,1) Squared Z');

%% 3. IC: HQIC
ICmatGARCH_t11=-2*LL_GARCHt11/T+2*(1+pbest+qbest+abest_t+gbest_t)*log(log(T))/T;
ICmatEGARCH_t11=-2*LL_EGARCHt11/T+2*(1+pbest+qbest+abest_t+gbest_t)*log(log(T))/T;
ICmatFIGARCH_t11=-2*LL_FIGARCHt11/T+2*(4)*log(log(T))/T;
ICmatAGARCH_t11=-2*LL_AGARCHt11/T+2*(6)*log(log(T))/T;
ICmatNAGARCH_t11=-2*LL_NAGARCHt11/T+2*(6)*log(log(T))/T;
ICmatGJRGARCH_t11=-2*LL_GJRGARCHt11/T+2*(6)*log(log(T))/T;

%% 4. Likelihood ratio test
[hEGARCH,pEGARCH,statEGARCH,cvEGARCH] = lratiotest(LL_EGARCHt11,LL_GARCHt11,1); % 1
[hFIGARCH,pFIGARCH,statFIGARCH,cvFIGARCH] = lratiotest(LL_FIGARCHt11,LL_GARCHt11,1);
[hAGARCH,pAGARCH,statAGARCH,cvAGARCH] = lratiotest(LL_AGARCHt11,LL_GARCHt11,1); % 1
[hNAGARCH,pNAGARCH,statNAGARCH,cvNAGARCH] = lratiotest(LL_NAGARCHt11,LL_GARCHt11,1);
[hGJRGARCH,pGJRGARCH,statGJRGARCH,cvGJRGARCH] = lratiotest(LL_GJRGARCHt11,LL_GARCHt11,1); % 1

%% 5. ARCH-LM Test
[lm_GARCH11, pval_lmGARCH11] = lmtest1(z_GARCH_t11,Lags);
[lm_EGARCH11, pval_lmEGARCH11] = lmtest1(z_EGARCH_t,Lags);
[lm_FIGARCH11, pval_lmFIGARCH11] = lmtest1(z_FIGARCH_t,Lags);
[lm_AGARCH11, pval_lmAGARCH11] = lmtest1(z_AGARCH_t,Lags);
[lm_NAGARCH11, pval_lmNAGARCH11] = lmtest1(z_NAGARCH_t,Lags);
[lm_GJRGARCH11, pval_lmGJRGARCH11] = lmtest1(z_GJRGARCH_t,Lags);

mean(pval_lmGJRGARCH11)
mean(pval_lmEGARCH11)
mean(pval_lmFIGARCH11)
mean(pval_lmAGARCH11)
mean(pval_lmNAGARCH11)
mean(pval_lmGJRGARCH11)

% Assuming dt is your time vector and V_GARCH_t11, V_EGARCHt11, V_FIGARCHt11, V_AGARCHt11, V_NAGARCHt11, V_GJRGARCHt11 are the corresponding data vectors for each model.

% Define the range of indices
startIndex = 1;
endIndex = min(2516, length(dt)); % Ensure endIndex does not exceed the length of dt

% Plotting the sliced data
plot(dt(startIndex:endIndex), [V_GARCH_t11(startIndex:endIndex), V_EGARCHt11(startIndex:endIndex), V_FIGARCHt11(startIndex:endIndex), V_AGARCHt11(startIndex:endIndex), V_NAGARCHt11(startIndex:endIndex), V_GJRGARCHt11(startIndex:endIndex)]);
legend({'GARCH','EGARCH','FIGARCH','AGARCH','NAGARCH','GJRGARCH'}, 'FontSize', 14);
xlabel('Time', 'FontSize', 14);
ylabel('Volatility', 'FontSize', 14);
title('Comparison of GARCH class', 'FontSize', 16);



% Define the range of indices
startIndex = 1;
endIndex = min(2516, length(dt)); % Ensure endIndex does not exceed the length of dt

% Plotting the sliced data as points
scatter(dt(startIndex:endIndex), [V_GARCH_t11(startIndex:endIndex), ...
    V_EGARCHt11(startIndex:endIndex), V_FIGARCHt11(startIndex:endIndex),...
    V_AGARCHt11(startIndex:endIndex), V_NAGARCHt11(startIndex:endIndex), ...
    V_GJRGARCHt11(startIndex:endIndex)], '+');
legend({'GARCH','EGARCH','FIGARCH','AGARCH','NAGARCH','GJRGARCH'}, 'FontSize', 14);
xlabel('Time', 'FontSize', 14);
ylabel('Volatility', 'FontSize', 14);
title('Comparison of GARCH Models', 'FontSize', 16);
            %%  d.1) ARMA-GARCH BEST Model Diagnosis %%

% refer to 673th-704th line

%% Volatility Estimation and Forecasting (used to compare RVs Models) %%

%plot forecasted conditional variance

figure
hold on
h1 = plot(outofsmpl,vf,'b','LineWidth',1);
h2 = plot(outofsmpl,vf2,'r','LineWidth',1);
h3 = plot(outofsmpl,vf3,'g','LineWidth',1);
h4 = plot(outofsmpl,vf4,'m','LineWidth',1);

legend([h1 h2 h3 h4], 'Cond. Var. Forecast: GARCH',...
		'Cond. Var. Forecast: GARCH (t-distr.)', ...
        'Cond. Var. Forecast: E-GARCH', ...
		'Cond. Var. Forecast: GJR-GARCH', 'Location','NorthWest');
title('Two-Year Forecasted Conditional Variance')
hold off

%Finally, the forecast() function can be used to forecast conditional mean
%and variance simultaneously in the same fashion as forecasting an ARMA
%model. For the general setting, see solutions to the previous workshop.

insmpl=dt(year(dt)<=2012);
ret_insmpl=ret(year(dt)<=2012);
outofsmpl=dt(year(dt)>2012);
ret_outofsmpl=ret(year(dt)>2012);

insmpl_estmdl=estimate(Mdl,ret);
insmpl_estmdl2=estimate(Mdl2,ret);
insmpl_estmdl3=estimate(Mdl3,ret);
insmpl_estmdl4=estimate(Mdl4,ret);

[y,ymse,vf]=forecast(insmpl_estmdl,length(outofsmpl),'Y0',ret_insmpl);
[y2,ymse2,vf2]=forecast(insmpl_estmdl2,length(outofsmpl),'Y0',ret_insmpl);
[y3,ymse3,vf3]=forecast(insmpl_estmdl3,length(outofsmpl),'Y0',ret_insmpl);
[y4,ymse4,vf4]=forecast(insmpl_estmdl4,length(outofsmpl),'Y0',ret_insmpl);

%%

%plot

figure
hold on
h1 = plot(outofsmpl,y,'b','LineWidth',1);
plot(outofsmpl,y + 1.96*sqrt(ymse),'b:','LineWidth',1);
plot(outofsmpl,y - 1.96*sqrt(ymse),'b:','LineWidth',1);

h3 = plot(outofsmpl,y2,'r','LineWidth',1);
plot(outofsmpl,y2 + 1.96*sqrt(ymse2),'r:','LineWidth',1);
plot(outofsmpl,y2 - 1.96*sqrt(ymse2),'r:','LineWidth',1);

h5 = plot(outofsmpl,y3,'g','LineWidth',1);
plot(outofsmpl,y3 + 1.96*sqrt(ymse3),'g:','LineWidth',1);
plot(outofsmpl,y3 - 1.96*sqrt(ymse3),'g:','LineWidth',1);

h7 = plot(outofsmpl,y4,'m','LineWidth',1);
plot(outofsmpl,y4 + 1.96*sqrt(ymse4),'m:','LineWidth',1);
plot(outofsmpl,y4 - 1.96*sqrt(ymse4),'m:','LineWidth',1);

h9 = plot(outofsmpl, ret_outofsmpl,'Color','k','LineWidth',0.1);

legend([h9 h1 h3 h5 h7], 'Observed', 'Forecast: GARCH',...
		'Forecast: GARCH (t-distr.)', 'Forecast: E-GARCH',...
		'Forecast: GJR-GARCH', 'Location','NorthWest');
title('Two-Year Forecasts and Approximate 95% Confidence Intervals')
hold off







%%               PART 2: Realised Volatility and Forecasting            %%
%%  e.1) %%

help MFEToolbox;
%If not: Home > Set Path > Add with subfolders... > 'MFEToolbox' folder (unzipped)
%OR use following line (Note you need to set your own folder path!)
%addpath(genpath('YOUR_FOLDER_PATH_TO_MFE_TOOLBOX_FOLDER\MFEToolbox'));
addpath(genpath('E:\OneDrive - Lancaster University\Msc MBF\2. AC.F609 Financial Econometrics\Crosswork_22nd March 2024\Teamwork\MFEToolbox')); %example
% set the numerical data
data = readtable('HPQ_trade_quote');
HPnumericData = table2array(data);
%Load the numericData data
load('numericData.mat');


%Day counter
dc=unique(HPnumericData(:,1));

%Extract unique dates
ddate=unique(HPnumericData(:,2));
%%
%get datetime array for plot
dt = datetime(num2str(ddate),'InputFormat','yyyyMMdd','Format','dd/MM/yyyy');

%Initialize tick return
ret = [];

%initialize tick rv
tickrv=zeros(length(dc),1);

%Set the time grid to compute the volatility signature plot up to 30 min
tgrid=[1, 5:5:2700];

%Initialize a matrix for  volalatility signature plot
VSP=zeros(length(dc),length(tgrid));

%initialize various measures of RV
rv=zeros(length(dc),1);
subrv=zeros(length(dc),1);
tsrv=zeros(length(dc),1);
rk=zeros(length(dc),1);
prerv=zeros(length(dc),1);
bpv=zeros(length(dc),1);
prebpv=zeros(length(dc),1);


%For loop for all trading days
for i=1:length(dc)
   HPnumericDataset=HPnumericData(HPnumericData(:,1)==dc(i),3:4);
   temp=diff(log(HPnumericDataset(:,2)));
   
   %Collecting tick return
   ret=[ ret; temp];
   
   %Computing tick-by-tick RV measure
   tickrv(i)=temp'*temp;  

%Using the MFEtoolbox to construct volatility signature plot based on the
%sampling frequencies in the tgrid
    for j=1:length(tgrid)
        RV=realized_variance(HPnumericDataset(:,2),HPnumericDataset(:,1),...
            'seconds','CalendarTime',tgrid(j));
        VSP(i,j)=RV;
    end

%Compute the 5-min simple RV and 5-min RV subsampled every 1min
    [RV,RVSS]=realized_variance(HPnumericDataset(:,2),HPnumericDataset(:,1),...
       'seconds','CalendarTime',3000,50);
    rv(i)=RV;
    subrv(i)=RVSS;

%Compute TSRV with the fast scale sampled every 1 second (noise variance
%estimator) and 5 second subsampling
    [~,~,~,RVTSSD]=realized_twoscale_variance(HPnumericDataset(:,2),HPnumericDataset(:,1),...
       'seconds','CalendarTime',10,50);
    tsrv(i)=RVTSSD;
%Compute 5-min RK with nonflat parzen kernel and automatic bandwith
%selection
    [~,RK]=realized_kernel(HPnumericDataset(:,2),HPnumericDataset(:,1),...
       'seconds','CalendarTime',3000);
    rk(i)=RK;
%Compute 5-min preaveraged RV
    [PR]=realized_preaveraged_variance(HPnumericDataset(:,2),HPnumericDataset(:,1),...
       'seconds','CalendarTime',3000);
    prerv(i)=PR;
%Compute 5-min realized (debiased) bipower variation
    [~,~,RBV]=realized_bipower_variation(HPnumericDataset(:,2),HPnumericDataset(:,1),...
       'seconds','CalendarTime',3000);
    bpv(i)=RBV;

% Corrected function call to match the function's output
prerbv = realized_preaveraged_bipower_variation(HPnumericDataset(:,2), HPnumericDataset(:,1),...
   'seconds', 'CalendarTime', 3000);
prebpv(i) = prerbv;



end




%%

%Analyse the properties of tick returns and tick RV
figure('units','normalized','outerposition',[0 0 1 1]);
subplot(2,2,1);
histfit(ret);
title('Histogram of Tick-by-Tick Returns');
subplot(2,2,2);
autocorr(ret,50);
title('Autocorrelogram of Tick-by-Tick Returns');
xlim([1,50]);
ylim([-0.15 ,0.02]);
subplot(2,2,3);
autocorr(ret.^2,100);
title('Autocorrelogram of Tick-by-Tick Squared Returns');
ylim([0 ,0.15]);
xlim([1,100]);
subplot(2,2,4)
autocorr(tickrv,100);
title('Autocorrelogram of Tick-by-Tick RV');
xlim([1,100]);

%%

%Plot the volatility signature plot
figure('units','normalized','outerposition',[0 0 1 1]);
plot(tgrid,mean(VSP)); % VSP averaged over days
title('Volatility Signature Plot');
xlabel('Sampling Interval In Seconds');
ylabel('Average RV');

%%

%Plot different RV measures
figure('units','normalized','outerposition',[0 0 1 1]);
plot(dt, [tickrv rv subrv tsrv rk prerv bpv]);
legend({'tick-by-tick RV' '5min RV' 'Subsampled RV' 'TSRV' 'RK' 'Preaveraged RV' 'RBPV'});


%%  f.1) %%

%%  g.1) %%
%for part2_task3/4
% Assuming 'ddate', 'tickRV', 'RV', 'subRV', 'TSRV', 'RK', 'preRV', 'RBV', and 'preBPV'
% are column vectors of the same length:

% Create a table with all the variables
RVTable_CW = table(ddate, tickrv, rv, subrv, tsrv, rk, prerv, bpv, prebpv);
RVTable_CW.tickrv = double(RVTable_CW.tickrv);
RVTable_CW.rv = double(RVTable_CW.rv);
RVTable_CW.subrv = double(RVTable_CW.subrv);
RVTable_CW.tsrv = double(RVTable_CW.tsrv);
RVTable_CW.rk = double(RVTable_CW.rk);
RVTable_CW.prerv = double(RVTable_CW.prerv);
RVTable_CW.bpv = double(RVTable_CW.bpv);
RVTable_CW.prebpv = double(RVTable_CW.prebpv);

RVtbl_CW = table2array(RVTable_CW);
save('RVtbl_CW.mat', 'RVTable_CW');

load('RVtbl_CW.mat');
%%
%for part2_task2
fig10 = figure('units','normalized','outerposition',[0 0 1 1]); % Create figure with properties

% Now plot directly into fig10
subplot(2,2,1);
histfit(ret);
title('Histogram of Tick-by-Tick Returns');

subplot(2,2,2);
autocorr(ret,50);
title('Autocorrelogram of Tick-by-Tick Returns');
xlim([1,50]);
ylim([-0.15 ,0.02]);

subplot(2,2,3);
autocorr(ret.^2,100);
title('Autocorrelogram of Tick-by-Tick Squared Returns');
ylim([0 ,0.15]);
xlim([1,100]);

subplot(2,2,4);
autocorr(tickRV,100);
title('Autocorrelogram of Tick-by-Tick RV');
xlim([1,100]);

% Save fig10, which now contains your plots
saveas(fig10,'properties of tick returns and tick RV.jpg');

%%
% Create fig11 with specified properties and retain the figure handle
fig11 = figure('units','normalized','outerposition',[0 0 1 1]);

% Plot directly into fig11
plot(tgrid, mean(VSP)); % Assuming tgrid and VSP are defined earlier in your code
title('Volatility Signature Plot');
xlabel('Sampling Interval In Seconds');
ylabel('Average RV');

% Save fig11, which now contains your plot
saveas(fig11, 'volatility signature plot.jpg');

%%
% Initialize fig12 with the desired properties directly
fig12 = figure('units','normalized','outerposition',[0 0 1 1]);

% Now, plot directly into fig12 without creating a new figure
plot(dt, [tickRV RV subRV TSRV RK preRV RBV]);
legend({'tick-by-tick RV', '5min RV', 'Subsampled RV', 'TSRV', 'RK', 'Preaveraged RV', 'BPV'}, 'Location', 'best');
title('Volatility Measures Comparison');
xlabel('Time');
ylabel('Realized Volatility');

% Save fig12, which now contains your plots
saveas(fig12, 'volatility measures comparison.jpg');



%Load the dataset of RV measures
load('RVtbl_CW.mat');

%Get datetime array
dt = datetime(num2str(RVTable_CW{:,1}),'InputFormat','yyyyMMdd','Format','dd/MM/yyyy');

%Get the names of all series
RVnames=RVTable_CW.Properties.VariableNames(2:9);

%We choose the benchmark model to be RK as the l.h.s. of a HAR model
RK=RVTable_CW.rk;

%We then  do rolling-window forecasts using different r.h.s. variables.
%We store all the variables in a matrix below
RVmat=RVTable_CW{:,2:9};

%Plot the RV measures
figure('units','normalized','outerposition',[0 0 1 1]);
plot(dt, RVTable_CW{:,2:9});
legend(RVnames);

%%

%Choose the length of the insample period
insmpl=2013;

%Initialize a matrix to collect the forecasts
res=zeros(length(RK)-insmpl, size(RVmat,2)* 2);

%Write a loop to compute the forecasts
for i=1:size(RVmat,2)
   res(:,((2*i)-1):(2*i))=HAR_frcst(RK, RVmat(:,i), insmpl);
end
%Note that for the matrix res, all the odd columns store the true value RK,
%and all the even colums store the forecasts from different models,
%in the same order as stored in the RVtbl_CW.mat

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

%%  h.1) %%

