function trajOut = integrateDerivate(trajIn)
    % Reconstructs the position and velocity from the velocity and position, respectively, using numerical integration and differentiation.
    nDims = size(trajIn.p, 2);
    d = zeros(size(trajIn.p));
    for k = 1:nDims
        d(:,k) = gradient(trajIn.p(:,k), trajIn.t);
    end
    tangent = d ./ max(sqrt(sum(d.^2, 2)), eps);

    v = trajIn.v(:) .* tangent;
    posRec = trajIn.p(1,:) + cumtrapz(trajIn.t, v);

    velRec = zeros(size(posRec));
    for k = 1:nDims
        velRec(:,k) = gradient(posRec(:,k), trajIn.t);
    end

    trajOut = trajIn;
    trajOut.p = posRec;
    trajOut.v = sqrt(sum(velRec.^2, 2));
end