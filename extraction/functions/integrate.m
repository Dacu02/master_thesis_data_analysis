function trajOut = integrate(trajIn)
%INTEGRATE Sostituisce Position con la posizione ottenuta integrando
%   Velocity lungo la direzione tangente della Position esistente
%   (cumtrapz). Velocity resta invariata.
%   Il gradiente di Position qui serve solo per la direzione (versore
%   tangente) — il modulo dello spostamento viene da Velocity, non
%   dalla Position corrente: una velocità scalare da sola non contiene
%   informazione di direzione, motivo per cui serve comunque una
%   traiettoria di riferimento da cui prenderla.
    nDims = size(trajIn.p, 2);
    d = zeros(size(trajIn.p));
    for k = 1:nDims
        d(:,k) = gradient(trajIn.p(:,k), trajIn.t);
    end
    tangent = d ./ max(sqrt(sum(d.^2, 2)), eps);

    v = trajIn.v(:) .* tangent;
    posRec = trajIn.p(1,:) + cumtrapz(trajIn.t, v);

    trajOut = trajIn;
    trajOut.p = posRec;
end