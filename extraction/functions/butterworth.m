function trajOut = butterworth(trajIn, order, cutoffHz, position)
    if nargin < 4
        position = false;
    end
    nyq = trajIn.f / 2;
    if ~(0 < cutoffHz && cutoffHz < nyq)
        error('butterworth:cutoff', 'cutoffHz must be in (0, %.2f) Hz.', nyq);
    end
    [b, a] = butter(order, cutoffHz / nyq, 'low');
    trajOut = trajIn;
    if position
        trajOut.p = filtfilt(b, a, trajIn.p);
    else
        trajOut.v = filtfilt(b, a, trajIn.v);
    end
end