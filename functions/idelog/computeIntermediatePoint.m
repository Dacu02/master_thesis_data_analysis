function [x, y, z] = computeIntermediatePoint(pStart, pEnd, u, v, startAngle, endAngle)
%COMPUTEINTERMEDIATEPOINT Punto medio dell'arco tra pStart e pEnd, dati
%   il piano (u,v) e gli angoli di inizio/fine dello stroke. Helper
%   privato di traj.idelog.
    a = u * cos(startAngle) + v * sin(startAngle);
    b = u * cos(endAngle) + v * sin(endAngle);
    normDiff = norm(b - a);
    if normDiff < 1e-12
        warning('computeIntermediatePoint:degenerateAngles', ...
            'startAngle e endAngle generano b==a: uso pStart come punto medio.');
        pMid = pStart;
    else
        R = norm(pEnd - pStart) / normDiff;
        center = pStart - R * a;
        thetaMid = startAngle + (endAngle - startAngle) / 2;
        pMid = center + R * (u * cos(thetaMid) + v * sin(thetaMid));
    end
    x = pMid(1); y = pMid(2); z = pMid(3);
end