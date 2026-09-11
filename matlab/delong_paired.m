function result = delong_paired(labels, first, second)
[aucs, covariance] = delong_covariance(labels, [first(:).'; second(:).']);
difference = aucs(1) - aucs(2);
variance = covariance(1, 1) + covariance(2, 2) - 2 * covariance(1, 2);
if ~isfinite(variance)
    probability = NaN;
    status = "insufficient_class_count";
elseif variance <= 1e-15
    if abs(difference) <= 1e-15
        probability = 1;
        status = "identical_auc_zero_variance";
    else
        probability = NaN;
        status = "zero_variance";
    end
else
    probability = erfc(abs(difference) / sqrt(2 * variance));
    status = "ok";
end
result = struct('auc_first', aucs(1), 'auc_second', aucs(2), 'auc_difference', difference, 'variance', variance, 'p_value', probability, 'status', status);
end
