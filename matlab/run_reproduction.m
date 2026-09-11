function run_reproduction(configFile, manifestFile, outputDirectory, pythonExecutable, resume)
if nargin < 5
    resume = false;
end
codeDirectory = fileparts(fileparts(mfilename('fullpath')));
configFile = char(java.io.File(configFile).getCanonicalPath());
manifestFile = char(java.io.File(manifestFile).getCanonicalPath());
outputDirectory = char(java.io.File(outputDirectory).getCanonicalPath());
pythonExecutable = char(java.io.File(pythonExecutable).getAbsolutePath());
assert(isfile(configFile) && isfile(manifestFile) && isfile(pythonExecutable));
assert(license('test', 'Statistics_Toolbox'), 'Statistics and Machine Learning Toolbox is required for SFS');
if ~isfolder(outputDirectory)
    mkdir(outputDirectory);
end
exchangeDirectory = tempname(outputDirectory);
mkdir(exchangeDirectory);
command = java.util.ArrayList();
argumentsList = {pythonExecutable, '-m', 'hemorrhage', 'run', '--config', configFile, '--manifest', manifestFile, '--output', outputDirectory};
if resume
    argumentsList{end + 1} = '--resume';
end
for index = 1:numel(argumentsList)
    command.add(java.lang.String(argumentsList{index}));
end
builder = java.lang.ProcessBuilder(command);
builder.directory(java.io.File(codeDirectory));
environment = builder.environment();
environment.put('HEMORRHAGE_MATLAB_EXCHANGE', exchangeDirectory);
builder.redirectErrorStream(true);
logFile = fullfile(outputDirectory, 'python_execution.log');
builder.redirectOutput(java.io.File(logFile));
process = builder.start();
cleanup = onCleanup(@() terminate_child(process));
fprintf('Python execution log: %s\n', logFile);
sfs_worker(exchangeDirectory, process);
status = process.waitFor();
if status ~= 0
    error('Python workflow failed. See %s\n%s', logFile, fileread(logFile));
end
fprintf('Completed: %s\n', outputDirectory);
end

function terminate_child(process)
if process.isAlive()
    process.destroy();
end
end
