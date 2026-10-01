function trajOut = bessel(trajIn, order, cutoffHz)
%   Bessel Low Pass Filter
    Fs = trajIn.f;
    nyq = Fs / 2;
    if ~(0 < cutoffHz && cutoffHz < nyq)
        error('bessel:cutoff', 'cutoffHz deve stare in (0, %.2f) Hz.', nyq);
    end
    wo = 2*pi*cutoffHz;
    [z, p, k] = besself(order, wo);
    [zd, pd, kd] = bilinear(z, p, k, Fs, cutoffHz);   % prewarping su cutoffHz
    [sos, g] = zp2sos(zd, pd, kd);

    trajOut = trajIn;
    trajOut.v = filtfilt(sos, g, trajIn.v);
end