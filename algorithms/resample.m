function trajOut = resample(trajIn, targetFs, method)
    % resample on uniform grid
    if nargin < 3
        method = 'pchip';
    end
    tOld = trajIn.t;
    tNew = (tOld(1) : 1/targetFs : tOld(end))';

    trajOut = trajIn;
    trajOut.t = tNew;
    trajOut.p = interp1(tOld, trajIn.p, tNew, method);
    trajOut.v = interp1(tOld, trajIn.v, tNew, method);
    trajOut.f = targetFs;
end