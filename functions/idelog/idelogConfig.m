function config = idelogConfig()
    % Parameters of idelog
    config.SpatialResolution = 0.0254;
    config.SamplingFrequency = [];   % [] = usa trajIn.Fs

    setUp = struct;
    setUp.Type_of_link_between_target_points = 1; % arc of circumference
    setUp.Type_of_Bell_Shaped_Function = 2; % lognormal
    setUp.Type_of_Reconstruction = 1; % speed + path for reconstruction
    setUp.Time_between_to_and_t1 = 0.5;
    setUp.SNRs_work_out_by_segments = 1;
    setUp.number_of_control_points = 3;
    if setUp.Type_of_link_between_target_points == 2
        setUp.number_of_control_points = 5;
    end
    % Numero minimo di punti tra minimi di velocità consecutivi, >=
    % number_of_control_points
    setUp.number_of_points_between_velocity_minima = setUp.number_of_control_points;
    % Tempo minimo tra minimi di velocità. Se incompatibile con
    % number_of_points_between_velocity_minima, la frequenza di
    % campionamento viene aumentata (se change_to_optimal_resolution=1).
    % /DEFAULT 0.04/
    setUp.time_between_velocity_minima = 0.04;
    setUp.ScriptStudio_smoothing = 1;
    setUp.method_to_work_out_velocity_minima = 0; % when 0 uses velocity minima, otherwise angle change
    setUp.DRAW = 0;

    % OTTIMIZZAZIONI --------------------------------------------------
    setUp.change_to_optimal_resolution = 1;
    setUp.multipivot_sensitivity = 1;
    setUp.Multi_Bell_Shaped = 0;
    setUp.number_of_splitted_functions = 1;
    setUp.Multi_Link = 0;
    setUp.number_of_speed_optimizations = 2;
    setUp.automatic_speed_optimizations = 1;
    setUp.number_of_trayectory_optimizations = 2;
    setUp.automatic_trayectory_optimizations = 1;
    setUp.percentage_of_refined_bell_shapes = 20;
    setUp.percentage_of_refined_links = 20;
    setUp.to_time_optimization = 1;

    % RICOSTRUZIONE DELLA TRAIETTORIA -----------------------------------
    setUp.fluency = 5;
    setUp.range_of_mean_stroke_time = [0.12 0.14];
    setUp.range_of_variance_of_stroke_time = [-0.01 0.01];
    setUp.range_of_overlap_between_bell_shapes = [0.1 0.3];
    setUp.range_of_variability_of_overlap = [-0.01 0.01];

    config.SetUp = setUp;
end