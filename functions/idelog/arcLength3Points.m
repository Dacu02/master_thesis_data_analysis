function D = arcLength3Points(P1, P2, P3)
%ARCLENGTH3POINTS Lunghezza dell'arco di circonferenza per tre punti 3D
%   (o della spezzata P1-P2-P3 se quasi allineati). Helper privato di
%   traj.idelog.
    a = norm(P2 - P3); b = norm(P1 - P3); c = norm(P1 - P2);
    Area = 0.5 * norm(cross(P2 - P1, P3 - P1));
    tol = 1e-9 * max([a b c])^2;
    if Area < tol
        D = a + c;
        return
    end
    R = (a * b * c) / (4 * Area);
    alpha = a^2 * (b^2 + c^2 - a^2);
    beta  = b^2 * (c^2 + a^2 - b^2);
    gamma = c^2 * (a^2 + b^2 - c^2);
    center = (alpha * P1 + beta * P2 + gamma * P3) / (alpha + beta + gamma);
    v1 = (P1 - center) / R; v2 = (P2 - center) / R; v3 = (P3 - center) / R;
    theta12 = acos(max(min(dot(v1, v2), 1), -1));
    theta23 = acos(max(min(dot(v2, v3), 1), -1));
    D = R * (theta12 + theta23);
end