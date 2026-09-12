addpath src/master_thesis_data_analysis/lib/
msgFolder = fullfile(pwd, 'src', 'unisa_acg_ros2', 'haptics');
ros2genmsg(msgFolder);
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
raw = loadRecordingFromBag(bagFile, topic_name);

sequences = {
    struct('Name', 'butterworth_10hz', 'Steps', {{@(t) butterworth(t,4,10)}})
    struct('Name', 'chebyshev_10hz',  'Steps', {{@(t) chebyshev(t,4,10,0.5)}})
    struct('Name', 'resample_then_bw', 'Steps', {{@(t) resample(t,RESAMPLING_FREQUENCY), @(t) butterworth(t,4,10)}}) % best
    struct('Name', 'bw_then_resample', 'Steps', {{@(t) butterworth(t,4,10), @(t) resample(t,RESAMPLING_FREQUENCY)}})
    struct('Name', 'bessel_10hz', 'Steps', {{@(t) bessel(t,4,10)}})
    struct('Name', 'resample_then_bessel', 'Steps', {{@(t) resample(t,RESAMPLING_FREQUENCY), @(t) bessel(t,4,10)}})
};

results = table();
for i = 1:numel(sequences)
    seq = sequences{i};

    processed = raw;
    for k = 1:numel(seq.Steps)
        processed = seq.Steps{k}(processed);
    end

    reconstructed = integrateDerivate(processed);
    if length(reconstructed.t) ~= length(raw.t)
        reference = resample(raw, reconstructed.f);
    else
        reference = raw;
    end
    metrics = computeMetrics(reference.v, reconstructed.v, seq.Name);

    results = [results; table(string(seq.Name), metrics.SNRdB, metrics.MSE, ...
        'VariableNames', {'Sequence','SNRdB','MSE'})]; %#ok<AGROW>
end

disp(results)

