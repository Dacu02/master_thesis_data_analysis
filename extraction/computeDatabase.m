addpath src/master_thesis_data_analysis/functions
addpath src/master_thesis_data_analysis/functions/idelog
addpath src/master_thesis_data_analysis/functions/util

DATABASE_TYPE = getDatabaseType();
OUTPUT_DATA_FOLDER = "lognormal_database";
if DATABASE_TYPE ~= "PATH" && DATABASE_TYPE ~= "FORCE"
    error('Database type must be "PATH" or "FORCE" for this script.');
end

% create the ROS2 message files if they don't exist
MSG_FOLDER = fullfile(pwd, 'src', 'unisa_acg_ros2', 'haptics');
RECORDING_FOLDER = fullfile(pwd, 'recording');
BUILD_ROOT = fullfile(pwd, 'build', 'ros2_msgs');
if ~exist(BUILD_ROOT, 'dir')
    mkdir(BUILD_ROOT);
end
ros2genmsg(MSG_FOLDER, BuildRoot=BUILD_ROOT);

% extract all the .db3 files in the recording folder
db3Files = dir(fullfile(RECORDING_FOLDER, '**', '*.db3'));
bagFile = fullfile(db3Files(1).folder, db3Files(1).name);
recordingFolder = fullfile(pwd, 'recording');

% read the YAML file to get the topic name
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
SMOOTHING = 0;
if DATABASE_TYPE == "PATH"
    IDELOG_PARAMETERS = struct( ...
        'SamplingFrequency', RESAMPLING_FREQUENCY, ...
        'ScriptStudio_smoothing', SMOOTHING, ...
        'timeBetween', 0.875 ...
    );
else % DATABASE_TYPE == "FORCE"
    IDELOG_PARAMETERS = struct( ...
        'SamplingFrequency', RESAMPLING_FREQUENCY, ...
        'ScriptStudio_smoothing', SMOOTHING ...
    );
end

% process each .db3 file
for fileIndex = 1:numel(db3Files)
    % input
    db3FilePath = fullfile(db3Files(fileIndex).folder, db3Files(fileIndex).name);
    trajectory = loadRecordingFromBag(db3FilePath, topic_name);

    % pipeline
    resampledTrajectory = resample(trajectory, RESAMPLING_FREQUENCY);
    trimmedTrajectory = filterBoundaryStrokes(resampledTrajectory);
    paddedTrajectory = padTrimmed(trimmedTrajectory, resampledTrajectory);
    filteredTrajectory = chebyshevIdelog(paddedTrajectory, true);


    % idelog
    try
        [reconstructedTrajectory, strokes, ~, ~, velocityApproached] = idelog(filteredTrajectory, struct('SamplingFrequency', filteredTrajectory.f, 'ScriptStudio_smoothing', SMOOTHING));
    catch ME
        warning('Idelog failed for file %s: %s', db3FilePath, ME.message);
        continue;
    end


    % interpolate the reference data
    reference.p = interp1(trajectory.t, trajectory.p, reconstructedTrajectory.t, 'pchip', 'extrap');
    reference.v = interp1(trajectory.t, trajectory.v, reconstructedTrajectory.t, 'pchip', 'extrap');
    reference.t = reconstructedTrajectory.t;
    reference.f = reconstructedTrajectory.f;

    metrics = compareTrajectories(reference, reconstructedTrajectory);%, "Comparison for file: " + db3Files(fileIndex).name);
    % print metrics
    fprintf('File: %s\n', db3FilePath);
    fprintf('SNR T: %f\n', metrics.SNR_T);
    fprintf('SNR V: %f\n', metrics.SNR_V);
    fprintf("Strokes: %d\n", numel(strokes));

    % output
    export_path = fullfile(pwd, OUTPUT_DATA_FOLDER, exportPath(db3FilePath, DATABASE_TYPE));
    export_path_folders = strsplit(export_path, filesep);
    export_path_last_folder = export_path_folders{end};

    n = numel(strokes);
    if n > 0
        startPts = vertcat(strokes.StartPoint);   % Nx3
        midPts   = vertcat(strokes.MidPoint);     % Nx3
        endPts   = vertcat(strokes.EndPoint);     % Nx3
    else
        startPts = zeros(0,3); midPts = zeros(0,3); endPts = zeros(0,3);
    end

    lognorm_table = table([strokes.Id]', [strokes.D]', [strokes.Mu]', [strokes.Sigma]', [strokes.To]', ...
        startPts(:,1), startPts(:,2), startPts(:,3), ...
        midPts(:,1), midPts(:,2), midPts(:,3), ...
        endPts(:,1), endPts(:,2), endPts(:,3), ...
        repmat(metrics.SNR_T, n, 1), repmat(metrics.SNR_V, n, 1), ...
        'VariableNames', {'stroke','D','mu','sigma','to','start_x','start_y','start_z','mid_x','mid_y','mid_z','end_x','end_y','end_z','snr_t','snr_v'});

    trajectory_table = table(reconstructedTrajectory.t, ...
        reconstructedTrajectory.p(:,1), reconstructedTrajectory.p(:,2), reconstructedTrajectory.p(:,3), ...
        reconstructedTrajectory.v, ...
        reference.p(:,1), reference.p(:,2), reference.p(:,3), reference.v, ...
        'VariableNames', {'time','rec_x','rec_y','rec_z','rec_v','ref_x','ref_y','ref_z','ref_v'});

    [export_dir, export_name] = fileparts(export_path);

    lognormal_strokes_file = fullfile(export_dir, export_name + "_logn.csv");
    reconstructedTrajectory_file = fullfile(export_dir, export_name + ".csv");

    mkdir(fileparts(lognormal_strokes_file));
    mkdir(fileparts(reconstructedTrajectory_file));

    writetable(lognorm_table, lognormal_strokes_file);
    writetable(trajectory_table, reconstructedTrajectory_file);

end