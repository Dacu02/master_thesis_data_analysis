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
    vrReconstructed = RecoiDeLog3D.velocity_reconstruted;
    
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

    bell   = ParamiDeLog3D.iDeLog_parameters_bell_functions{1};
    tx = RecoiDeLog3D.iDeLog_target_and_intermediate_x_points;
    ty = RecoiDeLog3D.iDeLog_target_and_intermediate_y_points;
    tz = RecoiDeLog3D.iDeLog_target_and_intermediate_z_points;
    thetaS = RecoiDeLog3D.iDeLog_start_angles(:);
    thetaE = RecoiDeLog3D.iDeLog_end_angles(:);

    % Layout: row k = virtual target tp_(k-1), col 1 = target, col 2 = library intermediate
    assert(size(tx,1) == nStrokes + 1, 'idelog:layout', ...
        'Expected %d rows in target/intermediate points, found %d.', ...
        nStrokes + 1, size(tx,1));

    for i = 1:nStrokes
        parsed   = cellfun(@str2double, tokens{i});
        strokeId = parsed(1);

        tpPrev = [tx(strokeId,1),   ty(strokeId,1),   tz(strokeId,1)];
        tpNext = [tx(strokeId+1,1), ty(strokeId+1,1), tz(strokeId+1,1)];
        pLib   = [tx(strokeId,2),   ty(strokeId,2),   tz(strokeId,2)];

        % Eq. (19): D = r * |theta_e - theta_s| on the arc tp_(j-1) -> tp_j
        [D, midPoint] = arcFromAngles(tpPrev, tpNext, ...
            thetaS(strokeId), thetaE(strokeId), ...
            RecoiDeLog3D.iDeLog_u_plane_vectors{strokeId}, ...
            RecoiDeLog3D.iDeLog_v_plane_vectors{strokeId}, pLib);

        % Tangent directions at the two ends of the arc, in 3D
        [tanS, tanE, conv, score] = strokeTangents(thetaS(strokeId), thetaE(strokeId), ...
            RecoiDeLog3D.iDeLog_u_plane_vectors{strokeId}, ...
            RecoiDeLog3D.iDeLog_v_plane_vectors{strokeId}, tpNext - tpPrev);
        if score < 0.99
            warning('idelog:tangentConvention', ...
                'Stroke %d: chord/bisector alignment = %.4f (convention %d).', ...
                strokeId, score, conv);
        end
        [zenS, aziS] = dirToSpherical(tanS);
        [zenE, aziE] = dirToSpherical(tanE);

        % The console prints D with 2 decimals, hence the 0.006 tolerance
        if abs(D - parsed(2)) > 0.006
            warning('idelog:amplitudeMismatch', ...
                'Stroke %d: D from Eq. (19) = %.4f, console D = %.4f.', ...
                strokeId, D, parsed(2));
        end

        strokes(strokeId).Id         = strokeId;
        strokes(strokeId).D          = D;
        strokes(strokeId).Mu         = bell(strokeId, 3);   % columns 2 and 3 are [sigma, mu]
        strokes(strokeId).Sigma      = bell(strokeId, 2);
        strokes(strokeId).To         = parsed(5);

        strokes(strokeId).StartPoint = tpPrev;              % virtual targets, not on the trajectory
        strokes(strokeId).MidPoint   = midPoint;
        strokes(strokeId).EndPoint   = tpNext;
        strokes(strokeId).UVector    = RecoiDeLog3D.iDeLog_u_plane_vectors{strokeId};
        strokes(strokeId).Vvector    = RecoiDeLog3D.iDeLog_v_plane_vectors{strokeId};

        strokes(strokeId).StartZenith  = zenS;
        strokes(strokeId).StartAzimuth = aziS;
        strokes(strokeId).EndZenith    = zenE;
        strokes(strokeId).EndAzimuth   = aziE;
        %strokes(strokeId).LibraryIntermediate = pLib;
    end
end

function [tanS, tanE, bestConv, bestScore] = strokeTangents(thetaS, thetaE, u, v, chord)
%STROKETANGENTS Unit tangent vectors at the start and end of a stroke arc.
%   The convention is chosen so that the bisector of the two tangents
%   is aligned with the chord (true for any circular arc).
    u = u(:) / norm(u); v = v(:) / norm(v);
    chord = chord(:) / norm(chord);
    basis = {u, v; -v, u};              % convention 1: cos*u + sin*v, 2: cos*(-v) + sin*u
    bestScore = -inf;
    for k = 1:2
        a = basis{k,1}; b = basis{k,2};
        ts = cos(thetaS) * a + sin(thetaS) * b;
        te = cos(thetaE) * a + sin(thetaE) * b;
        bis = ts + te;
        score = dot(bis / norm(bis), chord);
        if score > bestScore
            bestScore = score; bestConv = k;
            tanS = ts.'; tanE = te.';
        end
    end
end

function [zenith, azimuth] = dirToSpherical(d)
%DIRTOSPHERICAL Zenith (from +Z) and azimuth (atan2(y,x)) of a direction, in radians.
    d = d(:) / norm(d);
    zenith  = acos(max(min(d(3), 1), -1));
    azimuth = atan2(d(2), d(1));
end