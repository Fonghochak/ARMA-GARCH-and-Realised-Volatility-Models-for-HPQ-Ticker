function [info_table, IC, params, params_table] = ARMA_GARCH_optimal1(data, pmax, qmax, gmax, amax, dist, modelType)
    % Validate inputs
    narginchk(7, 7);
    validateattributes(data, {'numeric'}, {'column'});
    validateattributes([pmax, qmax, gmax, amax], {'numeric'}, {'integer', 'positive'});
    validatestring(dist, {'Gaussian', 't'});
    validatestring(modelType, {'GARCH', 'EGARCH', 'GJR-GARCH'});
    
    % Initialization and preprocessing steps
    T = length(data);
    if amax + gmax >= T / 10
        error('Too many parameters relative to the sample size. Reduce pmax, qmax, gmax, or amax.');
    end
    
    % Initialize matrices for information criteria
    ICmat = inf(pmax+1, qmax+1, gmax+1, amax+1, 3); % Last dimension for AIC, BIC, HQIC
    
    % Main optimization loop
    for p = 1:pmax
        for q = 1:qmax
            for g = 1:gmax
                for a = 1:amax
                    try
                        volModel = selectGarchModel(a, g, modelType);
                        if p == 0 && q == 0
                            Mdl = arima('Constant', mean(data), 'Variance', volModel);
                        else
                            Mdl = arima('ARLags', 1:p, 'MALags', 1:q, 'Constant', mean(data), 'Variance', volModel);
                        end
                        Mdl.Distribution = dist;
                        
                        % Estimate model and calculate log-likelihood
                        [~,~,logL] = estimate(Mdl, data, 'Display', 'off');
                        
                        % Apply penalty for zero-lag models
                        penalty = (p == 0 && q == 0 && g == 0 && a == 0) * log(T) * 0.5;
                        numParams = p + q + g + a + 1; % Adjust for variance parameter
                        IC = calcIC(logL - penalty, T, numParams);
                        ICmat(p+1, q+1, g+1, a+1, :) = IC;
                    catch
                        continue; % Skip combinations that fail
                    end
                end
            end
        end
    end
    
    % Extract optimal parameters and IC values
    [params, IC] = extractOptimalParams(ICmat);
    
    % Prepare output tables
    infoValues = {modelType; dist};
    info_table = table(infoValues, 'VariableNames', {'Info'});
    info_table.Properties.RowNames = {'ModelType', 'Distribution'};
    params_table = table([params.AIC; params.BIC; params.HQIC], 'VariableNames', {'IC_Values'});
    params_table.Properties.RowNames = {'AIC', 'BIC', 'HQIC'};
end

function vol = selectGarchModel(a, g, modelType)
    switch modelType
        case 'GARCH'
            vol = garch(a, g);
        case 'EGARCH'
            vol = egarch(a, g);
        case 'GJR-GARCH'
            vol = gjr(a, g);
        otherwise
            error('Unsupported GARCH model type.');
    end
end % Close selectGarchModel

function IC = calcIC(logL, T, numParams)
    aic = -2 * logL + 2 * numParams;
    bic = -2 * logL + log(T) * numParams;
    hqic = -2 * logL + 2 * log(log(T)) * numParams;
    IC = [aic, bic, hqic];
end % Close calcIC

function [params, IC] = extractOptimalParams(ICmat)
    [minAIC, aicIdx] = min(ICmat(:,:,:, :, 1), [], 'all', 'linear');
    [minBIC, bicIdx] = min(ICmat(:,:,:, :, 2), [], 'all', 'linear');
    [minHQIC, hqicIdx] = min(ICmat(:,:,:, :, 3), [], 'all', 'linear');

    [aicP, aicQ, aica, gicA] = ind2sub(size(ICmat(:,:,:,:,1)), aicIdx);
    [bicP, bicQ, bicG, bicA] = ind2sub(size(ICmat(:,:,:,:,2)), bicIdx);
    [hqicP, hqicQ, hqicG, hqicA] = ind2sub(size(ICmat(:,:,:,:,3)), hqicIdx);

    params = struct(...
        'AIC', [aicP-1, aicQ-1, aicG-1, aicA-1], ...
        'BIC', [bicP-1, bicQ-1, bicG-1, bicA-1], ...
        'HQIC', [hqicP-1, hqicQ-1, hqicG-1, hqicA-1]);

    IC = struct(...
        'AIC', minAIC, ...
        'BIC', minBIC, ...
        'HQIC', minHQIC);
end % Close extractOptimalParams
