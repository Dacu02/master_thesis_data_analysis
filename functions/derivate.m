function trajOut = derivate(trajIn)
%DERIVATE Sostituisce Velocity con il modulo della derivata numerica di
%   Position. Position resta invariata.
    nDims = size(trajIn.p, 2);
    d = zeros(size(trajIn.p));
    for k = 1:nDims
        d(:,k) = gradient(trajIn.p(:,k), trajIn.t);
    end

    trajOut = trajIn;
    trajOut.v = sqrt(sum(d.^2, 2));
end