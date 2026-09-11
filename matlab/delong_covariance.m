function [aucs, covariance] = delong_covariance(labels, predictions)
labels = labels(:);
assert(size(predictions, 2) == numel(labels));
assert(all(ismember(labels, [0; 1])));
assert(all(isfinite(predictions), 'all'));
nPositive = sum(labels == 1);
nNegative = sum(labels == 0);
assert(nPositive > 0 && nNegative > 0);
nModels = size(predictions, 1);
positiveComponents = zeros(nModels, nPositive);
negativeComponents = zeros(nModels, nNegative);
aucs = zeros(nModels, 1);
for index = 1:nModels
    positive = predictions(index, labels == 1).';
    negative = predictions(index, labels == 0);
    kernel = double(positive > negative) + 0.5 .* double(positive == negative);
    positiveComponents(index, :) = mean(kernel, 2).';
    negativeComponents(index, :) = mean(kernel, 1);
    aucs(index) = mean(kernel, 'all');
end
if min(nPositive, nNegative) < 2
    covariance = NaN(nModels);
else
    covariance = cov(positiveComponents.') ./ nPositive + cov(negativeComponents.') ./ nNegative;
end
end
