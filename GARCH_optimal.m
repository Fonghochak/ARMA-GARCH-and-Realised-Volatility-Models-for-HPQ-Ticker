function [gbest,abest,ICmat ] = GARCH_optimal( data, gmax, amax, arp, map, method)
%This function computes the optimal AR and MA lags for an ARMA(p,q) model
%based on information criteria.

%Inputs:
% data: T-by-1 vector of data to be modeled by an ARMA(p,q) model
% gmax: integer. Maximum GARCH lags considered in the model
% amax: integer. Maximum ARCH lags considered in the model
% arp: number of AR parameters in the mean equation
% map: number of MA parameters in the mean equation
% method: string. Methods to used for choosing the optimal model.
% Choice:   'AIC': Akaike Information Criterion,
%           'BIC': Bayesian Information Criterion
%           'HQIC': Hannan-Quinn Information Criterion

%Outputs:
%gbest: optimal GARCH lags
%abest: optimal ARCH lags.
%ICmat: Matrix of the chosen information criteria

T=length(data);

%Restrict the possible number of parameters:
if amax+1+gmax>=T/10
    error('Too many parameters. Reduce pmax and qmax');
end

ICmat=zeros(1+gmax,amax);

%start from GARCH(0,0), which is an ARMA(arp, map) model). Note that at
%least an ARCH term is required for a GARCH model.
for i=1:gmax+1
    for j=1:amax+1
        %If it is GARCH(0,0), treat it as an ARMA model
        if i*j==1
            Mdltemp=arima('ARLags',arp, 'MAlags', map);
            [~,~,logL,~] = estimate(Mdltemp,data,'Display','off');
        elseif i>1
            if j>1
                Mdltemp=arima('ARLags',arp, 'MAlags', map,'Variance',garch(i-1,j-1));
                [~,~,logL,~] = estimate(Mdltemp,data,'Display','off');
            else
                %If it is a GARCH(i,0) model (no lagged conditional variance),
                %return negative infinity as likelihood (invalid model)
                logL=-Inf;
            end
        end
                
        if strcmp(method,'AIC')
            ICmat(i,j)=-2*logL/T+2*(arp+map+i+j)/T;
        elseif strcmp(method,'BIC')
            ICmat(i,j)=-2*logL/T+(arp+map+i+j)*log(T)/T;
        elseif strcmp(method,'HQIC')
            ICmat(i,j)=-2*logL/T+2*(arp+map+i+j)*log(log(T))/T;
        else
            error('Unknown choice of information criterion');
        end
    end
end

%Find the location of the minimum information
[gbest,abest]=find(ICmat==min(min(ICmat)));
gbest=gbest-1;
abest=abest-1;

end

