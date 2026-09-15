addpath src/master_thesis_data_analysis/functions
addpath src/master_thesis_data_analysis/functions/idelog
addpath src/master_thesis_data_analysis/functions/util
MSG_FOLDER = fullfile(pwd, 'src', 'unisa_acg_ros2', 'haptics');
BUILD_ROOT = fullfile(pwd, 'build', 'ros2_msgs');
if ~exist(BUILD_ROOT, 'dir')
    mkdir(BUILD_ROOT);
end

ros2genmsg(MSG_FOLDER, BuildRoot=BUILD_ROOT);
bagFile = fullfile(pwd, 'recording', 'LUCIA_DE_LUCIA', 'GEOMETRIC2_SPIRAL_3DX', 'recording_0', 'recording_0_0.db3');
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

sequences = {
    %struct('Name', 'butterworth_10hz', 'Steps', {{@(t) butterworth(t,FILTER_ORDER,CUTOFF_FREQUENCY)}})
    %struct('Name', 'chebyshev_10hz',  'Steps', {{@(t) chebyshev(t,FILTER_ORDER,CUTOFF_FREQUENCY,RIPPLE_DB)}})
    %struct('Name', 'resample_bw', 'Steps', {{@(t) resample(t,RESAMPLING_FREQUENCY), @(t) butterworth(t,FILTER_ORDER,CUTOFF_FREQUENCY)}}) % best
    %struct('Name', 'bw_integrate', 'Steps', {{@(t) butterworth(t,FILTER_ORDER,CUTOFF_FREQUENCY), @(t) integrate(t)}}, ...
    %'IdelogConfig', struct('SetUp', struct('ScriptStudio_smoothing', 0)))
    %struct('Name', 'bw_integrate_smoothing', 'Steps', {{@(t) butterworth(t,FILTER_ORDER,CUTOFF_FREQUENCY), @(t) integrate(t)}})
    struct('Name', 'resample_bw_integrate_smoothing', 'Steps', {{@(t) resample(t,RESAMPLING_FREQUENCY), @(t) butterworth(t,FILTER_ORDER,CUTOFF_FREQUENCY), @(t) integrate(t)}})
    %struct('Name', 'resample_cheby_integrate_smoothing', 'Steps', {{@(t) resample(t,RESAMPLING_FREQUENCY), @(t) chebyshev(t,FILTER_ORDER,CUTOFF_FREQUENCY,RIPPLE_DB), @(t) integrate(t)}})
    %struct('Name', 'resample_cheby_integrate', 'Steps', {{@(t) resample(t,RESAMPLING_FREQUENCY), @(t) chebyshev(t,FILTER_ORDER,CUTOFF_FREQUENCY,RIPPLE_DB), @(t) integrate(t)}}, ...
    %    'IdelogConfig', struct('SetUp', struct('ScriptStudio_smoothing', 0)))
    %struct('Name', 'manual_smoother', 'Steps', {{@(t) resample(t,200),@(t) resample(t,200,'cubic'), @(t) chebyshev(t,FILTER_ORDER, CUTOFF_FREQUENCY,RIPPLE_DB), @(t) integrate(t)}}, ...
    %    'IdelogConfig', struct('SetUp', struct('ScriptStudio_smoothing', 0)))
    struct('Name', 'resample_bw_integrate', ...
        'Steps', {{@(t) resample(t,RESAMPLING_FREQUENCY), @(t) butterworth(t,FILTER_ORDER,CUTOFF_FREQUENCY), @(t) integrate(t)}}, ...
        'IdelogConfig', struct('SetUp', struct('ScriptStudio_smoothing', 0)))
    %struct('Name', 'bw_then_resample', 'Steps', {{@(t) butterworth(t,FILTER_ORDER,CUTOFF_FREQUENCY), @(t) resample(t,RESAMPLING_FREQUENCY)}})
    %struct('Name', 'bessel_10hz', 'Steps', {{@(t) bessel(t,FILTER_ORDER,CUTOFF_FREQUENCY)}})
    %struct('Name', 'resample_then_bessel', 'Steps', {{@(t) resample(t,RESAMPLING_FREQUENCY), @(t) bessel(t,FILTER_ORDER,CUTOFF_FREQUENCY)}})
};

results = table();
for i = 1:numel(sequences)
    seq = sequences{i};

    processed = raw;
    for k = 1:numel(seq.Steps)
        processed = seq.Steps{k}(processed);
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
    metrics = compareTrajectories(reference, reconstructed, seq.Name);
    
    
    results = [results; table(string(seq.Name), metrics.SNR_T, metrics.SNR_V, length(strokes), ...
        'VariableNames', {'Sequence','SNR_T','SNR_V', 'Strokes'})]; %#ok<AGROW>
end
disp(" ")
disp(results)

