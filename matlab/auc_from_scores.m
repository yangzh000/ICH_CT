function value = auc_from_scores(labels, scores)
labels = labels(:);
scores = scores(:);
assert(numel(labels) == numel(scores));
assert(all(isfinite(scores)));
assert(all(ismember(labels, [0; 1])));
positive = scores(labels == 1);
negative = scores(labels == 0);
assert(~isempty(positive) && ~isempty(negative));
kernel = double(positive > negative.') + 0.5 .* double(positive == negative.');
value = mean(kernel, 'all');
end
