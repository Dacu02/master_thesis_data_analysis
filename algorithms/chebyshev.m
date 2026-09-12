function trajOut = chebyshev(trajIn, order, cutoffHz, rippleDb)
    if nargin < 4
        rippleDb = 0.5;
    end
    nyq = trajIn.f / 2;
    if ~(0 < cutoffHz && cutoffHz < nyq)
        error('chebyshev:cutoff', 'cutoffHz must be in (0, %.2f) Hz.', nyq);
    end
    [b, a] = cheby1(order, rippleDb, cutoffHz / nyq, 'low');
    trajOut = trajIn;
    trajOut.v = filtfilt(b, a, trajIn.v);
end