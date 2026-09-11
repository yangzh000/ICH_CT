function result = binary_metrics(labels, probabilities, thresholds, alpha)
if nargin < 4
    alpha = 0.05;
end
labels = labels(:);
probabilities = probabilities(:);
assert(all(isfinite(probabilities)) && all(probabilities >= 0 & probabilities <= 1));
assert(numel(labels) == numel(probabilities));
assert(all(isfinite(thresholds)) && all(thresholds >= 0 & thresholds <= 1));
if isscalar(thresholds)
    thresholds = repmat(thresholds, numel(labels), 1);
end
assert(numel(thresholds) == numel(labels));
predicted = probabilities >= thresholds(:);
tp = sum(predicted & labels == 1);
tn = sum(~predicted & labels == 0);
fp = sum(predicted & labels == 0);
fn = sum(~predicted & labels == 1);
[auc, covariance] = delong_covariance(labels, probabilities.');
z = sqrt(2) * erfcinv(alpha);
result = struct('n', numel(labels), 'ht', sum(labels == 1), 'cs', sum(labels == 0), 'tp', tp, 'fn', fn, 'tn', tn, 'fp', fp, 'auc', auc);
if isfinite(covariance)
    result.auc_ci_low = max(0, auc - z * sqrt(max(0, covariance)));
    result.auc_ci_high = min(1, auc + z * sqrt(max(0, covariance)));
else
    result.auc_ci_low = NaN;
    result.auc_ci_high = NaN;
end
names = {'accuracy', 'sensitivity', 'specificity', 'ppv', 'npv'};
numerators = [tp + tn, tp, tn, tp, tn];
denominators = [numel(labels), tp + fn, tn + fp, tp + fp, tn + fn];
for index = 1:numel(names)
    denominator = denominators(index);
    if denominator == 0
        estimate = NaN;
        low = NaN;
        high = NaN;
    else
        estimate = numerators(index) / denominator;
        adjustment = 1 + z ^ 2 / denominator;
        center = (estimate + z ^ 2 / (2 * denominator)) / adjustment;
        distance = z * sqrt(estimate * (1 - estimate) / denominator + z ^ 2 / (4 * denominator ^ 2)) / adjustment;
        low = max(0, center - distance);
        high = min(1, center + distance);
    end
    result.(names{index}) = estimate;
    result.([names{index} '_ci_low']) = low;
    result.([names{index} '_ci_high']) = high;
end
end
