function [out, info] = rampBoundaryEdges(traj, opts)
%RAMPBOUNDARYEDGES  Force the speed to zero at the edges of a trajectory
%   with a quartic ramp, and re-derive the positions from the resulting
%   speed. Meant to run on the output of cutBoundaryStrokes, but works on
%   any traj with the same fields.
%
%   [out, info] = rampBoundaryEdges(traj)
%   [out, info] = rampBoundaryEdges(traj, Name, Value, ...)
%
%   Input traj (struct)
%       p  [N x 3]  position
%       v  [N x 1]  speed magnitude
%       t  [N x 1]  timestamps [s], strictly increasing
%       f  scalar   sampling frequency [Hz]
%
%   Output out (struct): same fields as traj, same N.
%       out.p = P(int out.v dt), where P is the arc-length parametrization
%       of the original path. The geometric path and its end points are
%       unchanged; only the timing along the path changes where the speed
%       was modified.
%   Output info (struct): edge modes, mask of synthetic samples,
%       consistency metrics.
%
%   Name-Value options (defaults)
%       smoothWin    (0.04)   Savitzky-Golay window for the local speed fit [s]
%       minRampSpeed (0.03)   an edge is left untouched if its smoothed
%                             speed is already <= this value (no ramp built)
%       rampDur      (0.10)   duration of the edge ramp [s]
%       fitWin       (0.05)   window for v, a estimation after the ramp [s]
%       maxPeakRatio (1.25)   ramp peak limit relative to replaced peak speed
%       maxScaleDev  (0.05)   warning threshold on the v/p path length mismatch
%
%   info.modeLeft / info.modeRight are one of:
%       'skipped (already at rest)'  smoothed edge speed <= minRampSpeed
%       'ramp'                       a valid quartic ramp was applied
%       'unchanged (no valid fit)'   all window scales failed validity checks

arguments
    traj (1,1) struct
    opts.smoothWin    (1,1) double {mustBePositive} = 0.04
    opts.minRampSpeed (1,1) double {mustBeNonnegative} = 0.03
    opts.rampDur      (1,1) double {mustBePositive} = 0.10
    opts.fitWin       (1,1) double {mustBePositive} = 0.05
    opts.maxPeakRatio (1,1) double {mustBePositive} = 1.25
    opts.maxScaleDev  (1,1) double {mustBePositive} = 0.05
end

%% 1. Input validation
p = traj.p;  v = traj.v(:);  t = traj.t(:);
N = numel(t);
assert(size(p,1) == N && size(p,2) == 3 && numel(v) == N, ...
    'p must be Nx3; v and t must have N elements.');
assert(all(diff(t) > 0), 't must be strictly increasing.');
fs = (N-1) / (t(end) - t(1));

%% 2. Smoothing (used for the near-rest check and the local v,a fit)
win = max(5, 2*round(opts.smoothWin*fs/2) + 1);
vs  = smoothdata(v, 'sgolay', win);

%% 3. Edge ramp (skipped on edges already close to rest)
% Over the first rampDur seconds at each edge, the speed is replaced by
% v(tau) = c2*u^2 + c3*u^3 + c4*u^4, u = tau/T, with
%   v(0) = 0, a(0) = 0        (rest at the edge)
%   v(T) = vc, a(T) = ac      (match the kept signal, estimated after T)
%   int v dtau = L            (arc length of p over the window)
% so the path length of the window is preserved. Windows of 1, 1.5, 2 and 3
% times rampDur are tried in order; if none gives a valid profile
% (non-negative, unimodal, peak below the limit), the speed is left
% unchanged at that edge. If the edge's smoothed speed is already
% <= minRampSpeed, no ramp is attempted at all.
pf     = fillmissing(fillmissing(p, 'previous'), 'next');
w      = max(5, round(opts.fitWin*fs));
scales = [1 1.5 2 3];
vOut   = v;
synthetic = false(N,1);

if vs(1) <= opts.minRampSpeed
    modeL = 'skipped (already at rest)';
else
    idx = 1:N;
    [vr, m, ok] = edgeRamp(t(idx) - t(1), v(idx), vs(idx), pf(idx,:), w, ...
        opts.rampDur, scales, opts.maxPeakRatio);
    if ok
        vOut(1:m)      = vr;
        synthetic(1:m) = true;
        modeL = 'ramp';
    else
        modeL = 'unchanged (no valid fit)';
    end
end

if vs(end) <= opts.minRampSpeed
    modeR = 'skipped (already at rest)';
else
    idx = N:-1:1;
    [vr, m, ok] = edgeRamp(t(N) - t(idx), v(idx), vs(idx), pf(idx,:), w, ...
        opts.rampDur, scales, opts.maxPeakRatio);
    if ok
        vOut(N:-1:N-m+1)      = vr;
        synthetic(N:-1:N-m+1) = true;
        modeR = 'ramp';
    else
        modeR = 'unchanged (no valid fit)';
    end
end

%% 4. Positions derived from speed
% v is a scalar speed, so its integral gives the arc length
% s(t) = int v dt. The pose is read on the original path P(s), which is the
% arc-length parametrization of p. Then |dp/dt| = |P'(s)| * ds/dt = v, and
% integrating v returns s, which maps back to p through P. If the path
% length of v differs from the path length of p, v is rescaled by a global
% factor so that both end points match. Missing positions are replaced by
% the last available one (pf).
[pk, vk, scale] = speedConsistentPositions(t, vOut, pf);
if isfinite(scale) && abs(scale - 1) > opts.maxScaleDev
    warning('rampBoundaryEdges:pathLength', ...
        'Path lengths of v and p differ by %.1f%%; v was rescaled.', ...
        100*abs(scale - 1));
end

%% 5. Output assembly and consistency check
out = struct('p', pk, 'v', vk, 't', t, 'f', fs);

% speedRelRmsErr:      rms(|dp/dt| - v) / max(v), central differences on t
% lengthRelErr:        |int v dt - polyline length of p| / int v dt
% maxShiftOutsideRamp: largest distance between derived and original
%                      positions on samples where v was not replaced
shift  = vecnorm(pk - pf, 2, 2);
speedP = vecnorm(centralDiff(pk, t), 2, 2);
lenV   = trapz(t, vk);
consistency = struct( ...
    'scale',               scale, ...
    'speedRelRmsErr',      sqrt(mean((speedP - vk).^2)) / max(vk), ...
    'lengthRelErr',        abs(lenV - sum(vecnorm(diff(pk), 2, 2))) / lenV, ...
    'maxShiftOutsideRamp', max([0; shift(~synthetic)]));

info = struct( ...
    'modeLeft',    modeL, ...
    'modeRight',   modeR, ...
    'synthetic',   synthetic, ...
    'consistency', consistency);
end

%% Local functions
function [pNew, vNew, scale] = speedConsistentPositions(t, v, p)
% Reparametrize the path of p by the arc length implied by v.
sp = [0; cumsum(vecnorm(diff(p), 2, 2))];    % arc length of the path
Sp = sp(end);
sv = cumtrapz(t, v);                         % arc length implied by v
Sv = sv(end);
if ~(Sp > 0 && Sv > 0)
    pNew = p;  vNew = v;  scale = NaN;       % static trajectory: nothing to do
    return
end
scale = Sp / Sv;                             % equals 1 if v and p agree
vNew  = scale * v;
sNew  = min(max(scale * sv, 0), Sp);
[spU, iu] = unique(sp, 'first');             % repeated positions -> repeated s
pNew  = interp1(spU, p(iu,:), sNew, 'pchip');
end

function vel = centralDiff(p, t)
% Derivative of p with respect to t, central differences on real timestamps.
vel = zeros(size(p));
vel(2:end-1,:) = (p(3:end,:) - p(1:end-2,:)) ./ (t(3:end) - t(1:end-2));
vel(1,:)   = (p(2,:) - p(1,:)) / (t(2) - t(1));
vel(end,:) = (p(end,:) - p(end-1,:)) / (t(end) - t(end-1));
end

function [vr, m, ok] = edgeRamp(tau, v, vs, pp, w, rampDur, scales, maxPeakRatio)
% Inputs are oriented from the edge inward: index 1 is the edge sample and
% tau(1) = 0. m is the number of samples in the ramp window (joint included).
n  = numel(tau);
s  = [0; cumsum(vecnorm(diff(pp), 2, 2))];   % cumulative arc length
vr = [];
m  = 0;
ok = false;
for sc = scales
    j = find(tau >= sc*rampDur, 1);
    if isempty(j) || n - j < 2
        break
    end
    idx      = j:min(n, j+w-1);
    [vc, ac] = boundaryKinematics(tau(idx) - tau(j), v(idx));
    [vrc, okc] = quarticRamp(tau(1:j), vc, ac, s(j), max([vs(1:j); vc]), maxPeakRatio);
    if okc
        vr = vrc;
        m  = j;
        ok = true;
        return
    end
end
end

function [vc, ac] = boundaryKinematics(tt, vv)
% Speed and acceleration at tt = 0 from a local quadratic fit
% (tt uses the real timestamps, relative to the joint sample).
[c, ~, mu] = polyfit(tt, vv, 2);
z0 = -mu(1) / mu(2);
vc = max(polyval(c, z0), 0);
ac = (2*c(1)*z0 + c(2)) / mu(2);
end

function [vr, ok] = quarticRamp(tau, vc, ac, L, peakRef, maxPeakRatio)
% Speed profile with v(0)=0, a(0)=0, v(T)=vc, a(T)=ac, integral = L.
T  = max(tau);
vr = zeros(size(tau));
ok = false;
if ~(T > 0 && isfinite(L) && L > 0 && isfinite(vc) && isfinite(ac))
    return
end
A = [1 1 1; 2 3 4; 1/3 1/4 1/5];
c = A \ [vc; ac*T; L/T];
prof = @(u) c(1)*u.^2 + c(2)*u.^3 + c(3)*u.^4;
vr = prof(tau / T);

g = prof(linspace(0, 1, 200));
d = diff(g);
d = d(abs(d) > 1e-9*max(abs(g)));
s = sign(d);
changes  = nnz(diff(s) ~= 0);
nonNeg   = all(g >= -1e-9*max(g));
unimodal = changes <= 1 && (changes == 0 || s(1) > 0);
peakOk   = max(g) <= maxPeakRatio * peakRef;
ok = nonNeg && unimodal && peakOk;
end