function result = sfs_select(X, y, trainFolds, validationFolds, maxFeatures, C, gammaSpec, randomSeed)
if nargin < 8
    randomSeed = 0;
end
rng(randomSeed, 'twister');
X = double(X);
y = double(y(:));
assert(size(X, 1) == numel(y));
assert(all(ismember(y, [0; 1])));
assert(numel(trainFolds) == numel(validationFolds));
assert(maxFeatures >= 1 && C > 0);
X(~isfinite(X)) = NaN;
maximum = min(floor(maxFeatures), size(X, 2));
chosen = zeros(1, 0);
remaining = 1:size(X, 2);
scores = zeros(1, maximum);
bestScore = -Inf;
bestColumns = zeros(1, 0);
for step = 1:maximum
    candidateScores = zeros(1, numel(remaining));
    for candidate = 1:numel(remaining)
        columns = [chosen, remaining(candidate)];
        foldScores = zeros(1, numel(trainFolds));
        for fold = 1:numel(trainFolds)
            train = double(trainFolds{fold}(:));
            validation = double(validationFolds{fold}(:));
            assert(isempty(intersect(train, validation)));
            trainingX = X(train, columns);
            validationX = X(validation, columns);
            medians = median(trainingX, 1, 'omitnan');
            medians(isnan(medians)) = 0;
            for column = 1:numel(columns)
                trainingX(isnan(trainingX(:, column)), column) = medians(column);
                validationX(isnan(validationX(:, column)), column) = medians(column);
            end
            mu = mean(trainingX, 1);
            sigma = std(trainingX, 1, 1);
            sigma(sigma == 0) = 1;
            trainingX = (trainingX - mu) ./ sigma;
            validationX = (validationX - mu) ./ sigma;
            if ischar(gammaSpec) || isstring(gammaSpec)
                if string(gammaSpec) == "scale"
                    variance = var(trainingX(:), 1);
                    if variance == 0
                        gamma = 1;
                    else
                        gamma = 1 / (numel(columns) * variance);
                    end
                elseif string(gammaSpec) == "auto"
                    gamma = 1 / numel(columns);
                else
                    error('Unknown SVM gamma option');
                end
            else
                gamma = double(gammaSpec);
            end
            assert(gamma > 0 && isfinite(gamma));
            model = fitcsvm(trainingX, y(train), 'KernelFunction', 'gaussian', 'KernelScale', 1 / sqrt(gamma), 'BoxConstraint', C, 'Standardize', false, 'ClassNames', [0; 1], 'Prior', 'empirical', 'Solver', 'SMO');
            [~, decision] = predict(model, validationX);
            positiveColumn = find(model.ClassNames == 1, 1);
            foldScores(fold) = auc_from_scores(y(validation), decision(:, positiveColumn));
        end
        candidateScores(candidate) = mean(foldScores);
    end
    [scores(step), winner] = max(candidateScores);
    chosen(end + 1) = remaining(winner);
    remaining(winner) = [];
    if scores(step) > bestScore + 1e-12
        bestScore = scores(step);
        bestColumns = chosen;
    end
end
result.selected_indices = bestColumns;
result.feature_order = chosen;
result.mean_auc_history = scores;
result.best_mean_auc = bestScore;
end
