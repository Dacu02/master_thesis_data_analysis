function [trajOut, strokes, snrT, snrV, velocityApproached, idelogObjectResult] = idelog(trajIn, configOverride)
%   Trajectory reconstruction using Sigma-Lognormal 3D (iDeLog3D).
%   [trajOut, strokes, snrT, snrV, velocityApproached] = IDELOG(trajIn)
%   uses default parameters (idelogDefaultConfig).
%   configOverride is a struct with the same fields as idelogDefaultConfig,
%   and can be used to override any of the default parameters.
%
%   trajOut is the reconstructed trajectory
%   strokes is an array of strokes, each stroke has (Id, Mu, Sigma, To, D, StartPoint, MidPoint, EndPoint). 
%   snrT/snrV are ParamiDeLog3D.SNR_*_reconstructed.
%   velocityApproached is RecoiDeLog3D.velocity_approached, the velocity used for reconstruction
    arguments
        trajIn (1,1) struct
        configOverride = []
    end

    defaults = idelogConfig();
    config = mergeStructs(defaults, configOverride);

    f = config.SamplingFrequency;
    if isempty(f)
        f = trajIn.f;
    end
    resol = config.SpatialResolution;
    setUp = config.SetUp;

    x = trajIn.p(:,1);
    y = trajIn.p(:,2);
    z = trajIn.p(:,3);

    
    figsBefore = findall(groot, 'Type', 'figure');
    previousVisibility = get(0, 'DefaultFigureVisible');
    cleanupObj = onCleanup(@() set(0, 'DefaultFigureVisible', previousVisibility)); 
    set(0, 'DefaultFigureVisible', 'off');

    [~, ~, ~, ~, ParamiDeLog3D] = iDeLog3D(x, y, z, f, resol, setUp);
    ttotalfirma = numel(y) / f;
    [xr, yr, ~, RecoiDeLog3D] = ReconstructiDeLog3D(ParamiDeLog3D, ttotalfirma, 0, 0);
    
    newFigs = setdiff(findall(groot, 'Type', 'figure'), figsBefore);
    close(newFigs);
    set(0, 'DefaultFigureVisible', 'on');
    idelogObjectResult = RecoiDeLog3D;
    zr = RecoiDeLog3D.z_reconstructed;
    vrReconstructed = RecoiDeLog3D.velocity_reconstruted;   % nome del toolbox, refuso incluso

    n = numel(xr);
    if numel(vrReconstructed) == n - 1
        vrReconstructed = [NaN; vrReconstructed(:)];
    elseif numel(vrReconstructed) ~= n
        error('idelog:velocityLength', ...
            'velocity_reconstruted has %d samples, expected %d or %d.', ...
            numel(vrReconstructed), n, n - 1);
    end

    n = numel(xr);
    fActual = (n - 1) / ttotalfirma;
    trajOut.p = [xr(:), yr(:), zr(:)];
    trajOut.v = vrReconstructed(:) * fActual;
    trajOut.t = (0:n-1)' / fActual + trajIn.t(1);
    trajOut.f = fActual;

    strokes = extractStrokes(ParamiDeLog3D, RecoiDeLog3D, ttotalfirma);

    snrT = ParamiDeLog3D.SNR_trajectory_reconstructed;
    snrV = ParamiDeLog3D.SNR_velocity_reconstructed;
    velocityApproached = RecoiDeLog3D.velocity_approached;
end

function strokes = extractStrokes(ParamiDeLog3D, RecoiDeLog3D, ttotalfirma) %#ok<INUSD>
%EXTRACTSTROKES Estrae D/mu/sigma/to e i punti geometrici per stroke.
%   "to" non è esposto in ParamiDeLog3D: l'unico modo noto è forzare
%   DRAW=1,WRITE=1 e leggerlo dalla console. I plot generati da questa
%   chiamata sono nascosti e chiusi sempre, indipendentemente da cosa
%   richiede il chiamante di traj.idelog, e la visibilità di default
%   viene ripristinata al termine anche in caso di errore.
    nStrokes = numel(RecoiDeLog3D.target_points_links_x_trajectory);
    
    figsBefore = findall(groot, 'Type', 'figure');
    previousVisibility = get(0, 'DefaultFigureVisible');
    cleanupObj = onCleanup(@() set(0, 'DefaultFigureVisible', previousVisibility)); 
    set(0, 'DefaultFigureVisible', 'off');

    consoleText = evalc('[~,~,~,~] = ReconstructiDeLog3D(ParamiDeLog3D, ttotalfirma, 1, 1);');
    
    newFigs = setdiff(findall(groot, 'Type', 'figure'), figsBefore);
    close(newFigs);
    set(0,'DefaultFigureVisible', 'on')
    %close all;

    num = '[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?';
    pattern = ['stroke:\s*(\d+)\s*,\s*D:\s*(' num ')\s*,\s*location\s*\(mu\):\s*(' num ...
               ')\s*,\s*scale\s*\(sigma\):\s*(' num ')\s*,\s*to:\s*(' num ')'];
    tokens = regexp(consoleText, pattern, 'tokens');
    if numel(tokens) ~= nStrokes
        error('idelog:strokeMismatch', ...
            'Found %d strokes in console text, expected %d.', numel(tokens), nStrokes);
    end

    strokes = repmat(struct('Id', 0, 'Mu', 0, 'Sigma', 0, 'To', 0, 'D', 0, ...
        'StartPoint', [0 0 0], 'MidPoint', [0 0 0], 'EndPoint', [0 0 0]), nStrokes, 1);


    position_minima_indexes = RecoiDeLog3D.position_minima_velocity_reconstructed;
    for i = 1:nStrokes
        parsed = cellfun(@str2double, tokens{i});
        strokeId = parsed(1);
        startPointIndex = position_minima_indexes(2*(strokeId-1) + 1);
        midPointIndex = position_minima_indexes(2*(strokeId-1) + 2);
        endPointIndex = position_minima_indexes(2*(strokeId-1) + 3);
        startPoint = [RecoiDeLog3D.x_reconstructed(startPointIndex), ...
                      RecoiDeLog3D.y_reconstructed(startPointIndex), ...
                      RecoiDeLog3D.z_reconstructed(startPointIndex)];
        midPoint = [RecoiDeLog3D.x_reconstructed(midPointIndex), ...
                    RecoiDeLog3D.y_reconstructed(midPointIndex), ...
                    RecoiDeLog3D.z_reconstructed(midPointIndex)];
        endPoint = [RecoiDeLog3D.x_reconstructed(endPointIndex), ...
                    RecoiDeLog3D.y_reconstructed(endPointIndex), ...
                    RecoiDeLog3D.z_reconstructed(endPointIndex)];
        
        %[midX, midY, midZ] = computeIntermediatePoint( ...
        %    startPoint, endPoint, ...
        %    -RecoiDeLog3D.iDeLog_v_plane_vectors{strokeId}', ...
        %    RecoiDeLog3D.iDeLog_u_plane_vectors{strokeId}', ...
        %    RecoiDeLog3D.iDeLog_start_angles(strokeId), ...
        %    RecoiDeLog3D.iDeLog_end_angles(strokeId));

        
        strokes(strokeId).Id = strokeId;
        % params
        strokes(strokeId).MidPoint = midPoint;
        strokes(strokeId).D = arcLength3Points(startPoint, strokes(strokeId).MidPoint, endPoint);
        strokes(strokeId).Mu = ParamiDeLog3D.iDeLog_parameters_bell_functions{1}(strokeId, 2);
        strokes(strokeId).Sigma = ParamiDeLog3D.iDeLog_parameters_bell_functions{1}(strokeId, 3);
        strokes(strokeId).To = parsed(5);
        
        % space
        strokes(strokeId).StartPoint = startPoint;
        strokes(strokeId).EndPoint = endPoint;
        strokes(strokeId).UVector = RecoiDeLog3D.iDeLog_u_plane_vectors;
        strokes(strokeId).Vvector = RecoiDeLog3D.iDeLog_v_plane_vectors;
        strokes(strokeId).StartAngle = RecoiDeLog3D.iDeLog_start_angles;
        strokes(strokeId).EndAngle = RecoiDeLog3D.iDeLog_end_angles;

        % space (idelog)
        strokes(strokeId).iDeLog_target_and_intermediate_x_points = RecoiDeLog3D.iDeLog_target_and_intermediate_x_points;
        strokes(strokeId).iDeLog_target_and_intermediate_y_points = RecoiDeLog3D.iDeLog_target_and_intermediate_y_points;
        strokes(strokeId).iDeLog_target_and_intermediate_z_points = RecoiDeLog3D.iDeLog_target_and_intermediate_z_points;
    end
end