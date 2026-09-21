addpath src/master_thesis_data_analysis/functions
addpath src/master_thesis_data_analysis/functions/idelog
addpath src/master_thesis_data_analysis/functions/util
MSG_FOLDER = fullfile(pwd, 'src', 'unisa_acg_ros2', 'haptics');
BUILD_ROOT = fullfile(pwd, 'build', 'ros2_msgs');
if ~exist(BUILD_ROOT, 'dir')
    mkdir(BUILD_ROOT);
end

ros2genmsg(MSG_FOLDER, BuildRoot=BUILD_ROOT);
bagFile = fullfile(pwd, 'recording', 'LUCIA_DE_LUCIA', 'GEOMETRIC2_SPIRAL_3DX', 'recording_4', 'recording_4_0.db3');
recordingFolder = fullfile(pwd, 'recording');

yamlFiles = dir(fullfile(recordingFolder, '**', '*.yaml'));
if isempty(yamlFiles)
    error('No YAML file found inside %s.', recordingFolder);
end
yaml_file = fullfile(yamlFiles(1).folder, yamlFiles(1).name);
yaml_text = fileread(yaml_file);

topic_name_match = regexp(yaml_text, '(?m)^\s*name:\s*([^\s#]+)', 'tokens', 'once');
if isempty(topic_name_match)
    error('No topic name found in %s.', yaml_file);
end
topic_name = topic_name_match{1};

RESAMPLING_FREQUENCY = 500;
CUTOFF_FREQUENCY = 14;
FILTER_ORDER = 4;
RIPPLE_DB = 0.01;
raw = loadRecordingFromBag(bagFile, topic_name);

% Each row contains: name, values, and the corresponding Idelog setup field.
% Add further rows to generate combinations with additional parameters.
parameterDefinitions = {
    'timeBetween', [0.85, 0.875, 0.9], 'Time_between_to_and_t1';
};

parameterNames = parameterDefinitions(:, 1);
parameterValues = parameterDefinitions(:, 2);
parameterFields = parameterDefinitions(:, 3);
parameterGrids = cell(size(parameterValues));
[parameterGrids{:}] = ndgrid(parameterValues{:});
numberOfSequences = numel(parameterGrids{1});
sequences = cell(1, numberOfSequences);

for sequenceIndex = 1:numberOfSequences
    setup = struct('ScriptStudio_smoothing', 0);
    nameParts = cell(1, numel(parameterNames));
    for parameterIndex = 1:numel(parameterNames)
        value = parameterGrids{parameterIndex}(sequenceIndex);
        setup.(parameterFields{parameterIndex}) = value;
        nameParts{parameterIndex} = sprintf('%s:%g', ...
            parameterNames{parameterIndex}, value);
    end

    sequences{sequenceIndex} = struct( ...
        'Name', ['resample_chebyshev_position_to_' strjoin(nameParts, '_')], ...
        'Steps', {{@(t) resample(t, RESAMPLING_FREQUENCY), @(t) chebyshevIdelog(t, true)}}, ...
        'IdelogConfig', struct('SetUp', setup));
end

results = table();
for i = 1:numel(sequences)
    seq = sequences{i};

    processed = raw;
    processedElements = cell(1, numel(seq.Steps) + 1);
    processedElements{1} = processed;
    for k = 1:numel(seq.Steps)
        processed = seq.Steps{k}(processed);
        processedElements{k + 1} = processed;
    end

    idelogOverride = struct('SamplingFrequency', processed.f);
    if isfield(seq, 'IdelogConfig')
        idelogOverride = mergeStructs(idelogOverride, seq.IdelogConfig);
    end

    [reconstructed, strokes, snrT, snrV, velocityApproached] = idelog(processed, idelogOverride);
    reference.p = interp1(raw.t, raw.p, reconstructed.t, 'pchip', 'extrap');
    reference.v = interp1(raw.t, raw.v, reconstructed.t, 'pchip', 'extrap');
    reference.t = reconstructed.t;
    reference.f = reconstructed.f;
    %reference = resample(reference, RESAMPLING_FREQUENCY);
    %reference = reference((length(reference.t) - length(reconstructed.t) + 1):end);
    metrics = compareTrajectories(reference, reconstructed);
    mean_D = mean([strokes.D]);
    for stroke_index = 1:length(strokes)
        if strokes(stroke_index).D < mean_D * 0.15
            warning('Stroke %d in sequence %s has D value %f, which is less than 15%% of the mean D value %f.', ...
                stroke_index, seq.Name, strokes(stroke_index).D, mean_D);
        end
    end
    plotTrajectoriesN([raw, processed, reconstructed], ["Raw", "Processed", "Reconstructed"], seq.Name);
    % Display the results for each sequence
    results = [results; table(string(seq.Name), metrics.SNR_T, metrics.SNR_V, length(strokes), sum([strokes.D] < mean_D * 0.15), ...
        'VariableNames', {'Sequence','SNR_T','SNR_V', 'Strokes', 'Discarded'})]; %#ok<AGROW>
    disp([strokes.D]);


end
disp(" ")
disp(results)