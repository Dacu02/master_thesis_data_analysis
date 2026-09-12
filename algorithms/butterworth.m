function trajOut = butterworth(trajIn, order, cutoffHz)
    nyq = trajIn.f / 2;
    if ~(0 < cutoffHz && cutoffHz < nyq)
        error('butterworth:cutoff', 'cutoffHz must be in (0, %.2f) Hz.', nyq);
    end
    [b, a] = butter(order, cutoffHz / nyq, 'low');
    trajOut = trajIn;
    trajOut.v = filtfilt(b, a, trajIn.v);
end