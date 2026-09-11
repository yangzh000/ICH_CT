function evaluate_predictions(predictionFile, outputDirectory, configFile)
if nargin < 3
    alpha = 0.05;
    thresholds = (0.01:0.01:0.99).';
else
    config = jsondecode(fileread(configFile));
    alpha = config.evaluation.alpha;
    thresholds = config.evaluation.dca_thresholds(:);
end
assert(all(thresholds > 0 & thresholds < 1));
if ~isfolder(outputDirectory)
    mkdir(outputDirectory);
end
options = detectImportOptions(predictionFile, 'TextType', 'string', 'VariableNamingRule', 'preserve');
options = setvartype(options, {'patient_id', 'model'}, 'string');
input = readtable(predictionFile, options);
required = {'patient_id', 'model', 'label', 'probability', 'threshold'};
assert(all(ismember(required, input.Properties.VariableNames)));
models = unique(input.model, 'sorted');
metricRows = cell(numel(models), 1);
aligned = cell(numel(models), 1);
curveRows = cell(numel(models), 1);
rocFigure = figure('Visible', 'off', 'Color', 'w');
rocAxes = axes(rocFigure);
hold(rocAxes, 'on');
dcaFigure = figure('Visible', 'off', 'Color', 'w');
dcaAxes = axes(dcaFigure);
hold(dcaAxes, 'on');
for index = 1:numel(models)
    group = sortrows(input(input.model == models(index), :), 'patient_id');
    assert(numel(unique(group.patient_id)) == height(group));
    if index > 1
        assert(isequal(group.patient_id, aligned{1}.patient_id));
        assert(isequal(group.label, aligned{1}.label));
    end
    aligned{index} = group;
    metric = binary_metrics(group.label, group.probability, group.threshold, alpha);
    metric.model = models(index);
    metricRows{index} = struct2table(metric);
    [~, order] = sort(group.probability, 'descend');
    sortedScores = group.probability(order);
    sortedLabels = group.label(order);
    ends = [find(diff(sortedScores) ~= 0); numel(sortedScores)];
    truePositive = cumsum(sortedLabels == 1) / sum(sortedLabels == 1);
    falsePositive = cumsum(sortedLabels == 0) / sum(sortedLabels == 0);
    plot(rocAxes, [0; falsePositive(ends)], [0; truePositive(ends)], 'LineWidth', 1.5, 'DisplayName', sprintf('%s (AUC %.3f)', models(index), metric.auc));
    netBenefit = zeros(numel(thresholds), 1);
    prevalence = mean(group.label);
    treatAll = prevalence - (1 - prevalence) .* thresholds ./ (1 - thresholds);
    for thresholdIndex = 1:numel(thresholds)
        positive = group.probability >= thresholds(thresholdIndex);
        tp = sum(positive & group.label == 1);
        fp = sum(positive & group.label == 0);
        netBenefit(thresholdIndex) = tp / height(group) - fp / height(group) * thresholds(thresholdIndex) / (1 - thresholds(thresholdIndex));
    end
    curveRows{index} = table(repmat(models(index), numel(thresholds), 1), thresholds, netBenefit, treatAll, zeros(numel(thresholds), 1), 'VariableNames', {'model', 'threshold', 'net_benefit', 'treat_all', 'treat_none'});
    plot(dcaAxes, thresholds, netBenefit, 'LineWidth', 1.5, 'DisplayName', models(index));
end
metrics = vertcat(metricRows{:});
writetable(metrics, fullfile(outputDirectory, 'external_metrics.csv'));
curves = vertcat(curveRows{:});
writetable(curves, fullfile(outputDirectory, 'decision_curves.csv'));
comparisons = cell(0, 1);
habitatIndex = find(models == "habitat", 1);
if ~isempty(habitatIndex)
    reference = aligned{habitatIndex};
    for index = 1:numel(models)
        if index == habitatIndex
            continue;
        end
        result = delong_paired(reference.label, reference.probability, aligned{index}.probability);
        result.first_model = "habitat";
        result.second_model = models(index);
        comparisons{end + 1, 1} = struct2table(result);
    end
end
if ~isempty(comparisons)
    writetable(vertcat(comparisons{:}), fullfile(outputDirectory, 'delong_comparisons.csv'));
end
plot(rocAxes, [0, 1], [0, 1], '--', 'Color', [0.6, 0.6, 0.6], 'HandleVisibility', 'off');
xlabel(rocAxes, 'False-positive rate');
ylabel(rocAxes, 'True-positive rate');
xlim(rocAxes, [0, 1]);
ylim(rocAxes, [0, 1]);
legend(rocAxes, 'Location', 'southeast', 'Box', 'off');
plot(dcaAxes, thresholds, curves.treat_all(1:numel(thresholds)), '--', 'Color', [0.5, 0.5, 0.5], 'DisplayName', 'Treat all');
yline(dcaAxes, 0, 'k-', 'DisplayName', 'Treat none');
xlabel(dcaAxes, 'Threshold probability');
ylabel(dcaAxes, 'Net benefit');
ylim(dcaAxes, [min(-0.1, min(curves.net_benefit) - 0.02), 1]);
legend(dcaAxes, 'Location', 'best', 'Box', 'off');
exportgraphics(rocFigure, fullfile(outputDirectory, 'external_roc.pdf'), 'ContentType', 'vector');
exportgraphics(dcaFigure, fullfile(outputDirectory, 'external_decision_curve.pdf'), 'ContentType', 'vector');
close(rocFigure);
close(dcaFigure);
save(fullfile(outputDirectory, 'evaluation.mat'), 'metrics', 'curves', 'alpha');
end
