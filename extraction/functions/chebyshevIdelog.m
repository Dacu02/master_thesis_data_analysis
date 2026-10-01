function trajOut = chebyshevIdelog(trajIn, position, order, cutoffHz, stopbandDb)
    arguments
        trajIn     (1,1) struct
        position   (1,1) logical = true
        order      (1,1) double = 11
        cutoffHz   (1,1) double = 16
        stopbandDb (1,1) double = 80
    end
    nyq = trajIn.f / 2;
    if ~(0 < cutoffHz && cutoffHz < nyq)
        error('chebyshev2:cutoff', 'cutoffHz must be in (0, %.2f) Hz.', nyq);
    end
    [b, a] = cheby2(order, stopbandDb, cutoffHz / nyq, 'low');
    trajOut = trajIn;
    if position
        trajOut.p = filtfilt(b, a, trajIn.p);
    else
        trajOut.v = filtfilt(b, a, trajIn.v);
    end
end