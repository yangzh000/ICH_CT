function sfs_worker(exchangeDirectory, process)
if nargin < 2
    process = [];
end
assert(license('test', 'Statistics_Toolbox'), 'Statistics and Machine Learning Toolbox is required for SFS');
ready = fopen(fullfile(exchangeDirectory, 'READY'), 'w');
fprintf(ready, '%s', version);
fclose(ready);
while ~isfile(fullfile(exchangeDirectory, 'STOP')) && (isempty(process) || process.isAlive())
    requests = dir(fullfile(exchangeDirectory, 'request_*.mat'));
    if isempty(requests)
        pause(0.05);
        continue;
    end
    for index = 1:numel(requests)
        request = fullfile(requests(index).folder, requests(index).name);
        responseName = strrep(requests(index).name, 'request_', 'response_');
        response = fullfile(exchangeDirectory, responseName);
        temporary = fullfile(exchangeDirectory, ['pending_' responseName]);
        try
            input = load(request);
            if isfield(input, 'operation') && string(input.operation) == "evaluate"
                evaluate_predictions(input.predictionFile, input.outputDirectory, input.configFile);
                result = struct('status', 'complete');
            else
                result = sfs_select(input.X, input.y, input.trainFolds, input.validationFolds, input.maxFeatures, input.C, input.gammaSpec, input.randomSeed);
            end
        catch exception
            result = struct('error_message', getReport(exception, 'extended', 'hyperlinks', 'off'));
        end
        save(temporary, '-struct', 'result', '-v7');
        movefile(temporary, response, 'f');
        delete(request);
    end
end
end
