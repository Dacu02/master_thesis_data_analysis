addpath src/master_thesis_data_analysis/functions
addpath src/master_thesis_data_analysis/functions/idelog
addpath src/master_thesis_data_analysis/functions/util
MSG_FOLDER = fullfile(pwd, 'src', 'unisa_acg_ros2', 'haptics');
BUILD_ROOT = fullfile(pwd, 'build', 'ros2_msgs');
if ~exist(BUILD_ROOT, 'dir')
    mkdir(BUILD_ROOT);
end

ros2genmsg(MSG_FOLDER, BuildRoot=BUILD_ROOT);
bagFile = fullfile(pwd, 'recording', 'NOEMI_BIANCAMANO', 'GEOMETRIC1_SPIRAL_3DY', 'recording_12', 'recording_12_0.db3');
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


p_k = loadRecordingFromBag(bagFile, topic_name);

p_tk = resample(p_k, RESAMPLING_FREQUENCY);

r_k = chebyshevIdelog(p_tk, true);
r_kv = chebyshevIdelog(p_tk, false);

r_kbutt = butterworth(p_tk, FILTER_ORDER, CUTOFF_FREQUENCY);
[reconstructed_smoothed, strokes, snrT, snrV, velocityApproached] = idelog(r_k, struct('SetUp', struct('ScriptStudio_smoothing', 1)));
[reconstructed, strokes, snrT, snrV, velocityApproached] = idelog(r_k, struct('SetUp', struct('ScriptStudio_smoothing', 0)));
%plotTrajectoriesN([p_k, p_tk, r_k], ["p_k: Raw", "p_tk: Resampled", "r_k: Filtered"], "Trajectory Comparison");
plotTrajectoriesN([p_k, r_k], ["p_k: Raw", "r_k: Filtered"], "Trajectory Comparison");
plotTrajectoriesN([p_k, r_kv], ["p_k: Raw", "r_kv: Filtered"], "Trajectory Comparison");

%[reconstructed_integrated_smoothed, strokes, snrT, snrV, velocityApproached] = idelog(integrate(r_kv), struct('SetUp', struct('ScriptStudio_smoothing', 1)));
%[reconstructed_integrated, strokes, snrT, snrV, velocityApproached] = idelog(integrate(r_kv), struct('SetUp', struct('ScriptStudio_smoothing', 0)));
%plotTrajectoriesN([p_k, reconstructed_integrated_smoothed, reconstructed_integrated], ["p_K", "Reconstructed Integrated Smoothed", "Reconstructed Integrated"], "Reconstruction Comparison");