function [D, M, R] = arcFromAngles(Ps, Pe, thetaS, thetaE, u, v, refPoint)
%ARCFROMANGLES Arc length and midpoint of the circular arc Ps -> Pe in 3D.
%   refPoint is only used to choose on which side of the chord the arc bulges.
    Ps = Ps(:).'; Pe = Pe(:).'; refPoint = refPoint(:).';
    half  = (thetaE - thetaS) / 2;
    chord = Pe - Ps;
    c     = norm(chord);
    C     = Ps + chord / 2;

    if abs(half) < 1e-8                     % straight-line limit
        D = c; R = Inf; M = C;
        return
    end

    R = c / (2 * abs(sin(half)));
    D = R * abs(thetaE - thetaS);           % Eq. (19)

    n = cross(u(:), v(:)); n = n / norm(n);
    b = cross(chord / c, n.'); b = b / norm(b);

    sagitta = R * (1 - cos(half));
    cand = [C + sagitta * b; C - sagitta * b];
    [~, k] = min(vecnorm(cand - refPoint, 2, 2));
    M = cand(k, :);
