function padded = padTrimmed(trimmed, original)
    % padTrimmed Restores a trimmed trajectory to the sampling grid of the
    % original one. Outside the kept interval the speed is zero and the
    % position is held at the nearest kept end point, so the padded
    % trajectory stays consistent (p is the integral of v).
    %
    % If trimmed extends past the end of original, the extra samples are
    % discarded.
    %
    % Inputs:
    %   trimmed  - struct with p (Nkx3), v (Nkx1), t (Nkx1), f, where t is a
    %              contiguous subset of original.t (may exceed the end)
    %   original - struct with p (Nx3), v (Nx1), t (Nx1), f
    % Output:
    %   padded   - struct with the fields of original, N samples, t = original.t

    tOrig = original.t(:);
    tTrim = trimmed.t(:);
    N  = numel(tOrig);
    Nk = numel(tTrim);

    % Locate the kept interval on the original time grid
    [dStart, iL] = min(abs(tOrig - tTrim(1)));
    tol = 0.01 / original.f;   % 1% of the sampling period
    assert(dStart < tol, ...
        'trimmed.t does not start on the original time grid.');

    % Truncate trimmed if it extends beyond the end of original
    iR = iL + Nk - 1;
    if iR > N
        nKeep = N - iL + 1;
        trimmed.t = tTrim(1:nKeep);
        trimmed.p = trimmed.p(1:nKeep, :);
        trimmed.v = trimmed.v(1:nKeep, :);
        iR = N;
    end

    % Sanity check on the aligned end
    assert(abs(tOrig(iR) - trimmed.t(end)) < tol, ...
        'trimmed.t is not aligned with original.t.');

    padded = original;

    padded.v = zeros(N, 1);
    padded.v(iL:iR) = trimmed.v(:);

    padded.p = zeros(N, 3);
    padded.p(iL:iR, :)   = trimmed.p;
    padded.p(1:iL-1, :)  = repmat(trimmed.p(1, :),   iL-1, 1);
    padded.p(iR+1:N, :)  = repmat(trimmed.p(end, :), N-iR,  1);
end