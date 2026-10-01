function [out, info] = filterBoundaryStrokes(traj, opts)
%FILTERBOUNDARYSTROKES  Remove low-intensity strokes at both ends of a
%   trajectory, force the speed to zero at the new edges, and re-derive the
%   positions from the resulting speed.
%
%   [out, info] = filterBoundaryStrokes(traj)
%   [out, info] = filterBoundaryStrokes(traj, Name, Value, ...)
%
%   Input traj (struct)
%       p  [N x 3]  position
%       v  [N x 1]  speed magnitude
%       t  [N x 1]  timestamps [s], strictly increasing
%       f  scalar   sampling frequency [Hz]
%
%   Output out (struct): same fields as traj (N can decrease).
%       out.v is the modified speed; out.p = P(int out.v dt), where P is the
%       arc-length parametrization of the original path. The geometric path
%       and its end points are unchanged; only the timing along the path
%       changes where the speed was modified.
%   Output info (struct): strokes table, threshold, cut indices, edge modes,
%       mask of synthetic samples (output indexing), consistency metrics.
%
%   Name-Value options (defaults)
%       smoothWin    (0.04)   Savitzky-Golay window for detection only [s]
%       promK        (3)      minima prominence: promK * residual noise sigma
%       promFrac     (0.02)   minima prominence floor: fraction of peak speed
%       restPercent  (10)     lowest speed percentile used as noise floor
%       thrK         (3)      stroke length threshold = thrK*noiseSpeed*refDur
%       refDur       (0.15)   reference stroke duration [s]
%       lenThr       ([])     absolute stroke length threshold, overrides auto
%       relVmax      (0.20)   strokes with vmax below relVmax*max(vmax) are weak
%       maxCut       (5)      maximum strokes removed per side
%       edgeFrac     ([0 0])  [left right] move each edge inward until the
%                             speed reaches edgeFrac*vmax of the edge stroke
%       rampDur      (0.10)   duration of the edge ramp [s]
%       fitWin       (0.05)   window for v, a estimation after the ramp [s]
%       maxPeakRatio (1.25)   ramp peak limit relative to replaced peak speed
%       maxScaleDev  (0.05)   warning threshold on the v/p path length mismatch

arguments
    traj (1,1) struct
    opts.smoothWin    (1,1) double {mustBePositive} = 0.04
    opts.promK        (1,1) double {mustBePositive} = 3
    opts.promFrac     (1,1) double {mustBeNonnegative} = 0.02
    opts.restPercent  (1,1) double {mustBePositive} = 10
    opts.thrK         (1,1) double {mustBePositive} = 3
    opts.refDur       (1,1) double {mustBePositive} = 0.15
    opts.lenThr       double = []
    opts.relVmax      (1,1) double {mustBeNonnegative} = 0.20
    opts.maxCut       (1,1) double {mustBeInteger, mustBeNonnegative} = 5
    opts.edgeFrac     (1,2) double {mustBeNonnegative} = [0 0]
    opts.rampDur      (1,1) double {mustBePositive} = 0.10
    opts.fitWin       (1,1) double {mustBePositive} = 0.05
    opts.maxPeakRatio (1,1) double {mustBePositive} = 1.25
    opts.maxScaleDev  (1,1) double {mustBePositive} = 0.05
end

%% 1. Input validation
% The effective rate is computed from the timestamps, since f can deviate
% from the nominal value.
p = traj.p;  v = traj.v(:);  t = traj.t(:);
N = numel(t);
assert(size(p,1) == N && size(p,2) == 3 && numel(v) == N, ...
    'p must be Nx3; v and t must have N elements.');
assert(all(diff(t) > 0), 't must be strictly increasing.');
fs = (N-1) / (t(end) - t(1));

%% 2. Smoothing and noise estimate (used for detection only)
% The smoothed speed is used to find minima; strokes are measured on v.
win   = max(5, 2*round(opts.smoothWin*fs/2) + 1);
vs    = smoothdata(v, 'sgolay', win);
sigma = robustSigma(v - vs);
prom  = max(opts.promK*sigma, opts.promFrac*max(vs));

%% 3. Stroke segmentation from speed minima
% The first and last samples are boundaries, so partial strokes at the
% edges are counted as strokes.
b    = unique([1; find(islocalmin(vs, 'MinProminence', prom)); N]);
K    = numel(b) - 1;
dur  = t(b(2:end)) - t(b(1:end-1));
len  = zeros(K,1);
vmax = zeros(K,1);
for k = 1:K
    seg     = b(k):b(k+1);
    len(k)  = trapz(t(seg), v(seg));     % path length, sigma-lognormal D
    vmax(k) = max(v(seg));
end

%% 4. Stroke intensity threshold
% A stroke is weak if its length is below the path length that noise-level
% speed would produce over refDur, or if its peak speed is below relVmax
% times the largest stroke peak.
sv         = sort(vs);
vlow       = sv(1:max(5, ceil(N*opts.restPercent/100)));
noiseSpeed = median(vlow) + robustSigma(vlow);
if isempty(opts.lenThr)
    lenThr = opts.thrK * noiseSpeed * opts.refDur;
else
    lenThr = opts.lenThr;
end
weak = (len < lenThr) | (vmax < opts.relVmax * max(vmax));

%% 5. Edge trimming
% Strokes are removed from each side until one exceeds the threshold, up to
% maxCut per side, always keeping at least one stroke. Removed strokes are
% cut together with their positions.
nL = 0;
while nL < opts.maxCut && nL < K-1 && weak(nL+1)
    nL = nL + 1;
end
nR = 0;
while nR < opts.maxCut && nL + nR < K-1 && weak(K-nR)
    nR = nR + 1;
end
iL = b(nL+1);          % first kept sample
iR = b(K+1-nR);        % last kept sample

% Optional: move each edge inward along the flank of the edge stroke.
while opts.edgeFrac(1) > 0 && iL < iR && vs(iL) < opts.edgeFrac(1) * vmax(nL+1)
    iL = iL + 1;
end
while opts.edgeFrac(2) > 0 && iR > iL && vs(iR) < opts.edgeFrac(2) * vmax(K-nR)
    iR = iR - 1;
end

%% 6. Edge ramp (applied also when no stroke was removed)
% Over the first rampDur seconds at each edge, the speed is replaced by
% v(tau) = c2*u^2 + c3*u^3 + c4*u^4, u = tau/T, with
%   v(0) = 0, a(0) = 0        (rest at the edge)
%   v(T) = vc, a(T) = ac      (match the kept signal, estimated after T)
%   int v dtau = L            (arc length of p over the window)
% so the path length of the window is preserved. Windows of 1, 1.5, 2 and 3
% times rampDur are tried in order; if none gives a valid profile
% (non-negative, unimodal, peak below the limit), the speed is left
% unchanged at that edge.
pf     = fillmissing(fillmissing(p, 'previous'), 'next');
w      = max(5, round(opts.fitWin*fs));
scales = [1 1.5 2 3];
vOut   = v;
synthetic = false(N,1);
modeL  = 'unchanged';
modeR  = 'unchanged';

% Left edge: samples iL, iL+1, ..., iR (time runs forward).
idx = iL:iR;
[vr, m, ok] = edgeRamp(t(idx) - t(iL), v(idx), vs(idx), pf(idx,:), w, ...
    opts.rampDur, scales, opts.maxPeakRatio);
if ok
    vOut(iL:iL+m-1)      = vr;
    synthetic(iL:iL+m-1) = true;
    modeL = 'ramp';
end

% Right edge: samples iR, iR-1, ..., iL (time runs backwards).
idx = iR:-1:iL;
[vr, m, ok] = edgeRamp(t(iR) - t(idx), v(idx), vs(idx), pf(idx,:), w, ...
    opts.rampDur, scales, opts.maxPeakRatio);
if ok
    vOut(iR:-1:iR-m+1)      = vr;
    synthetic(iR:-1:iR-m+1) = true;
    modeR = 'ramp';
end

%% 7. Positions derived from speed
% v is a scalar speed, so its integral gives the arc length
% s(t) = int v dt. The pose is read on the original path P(s), which is the
% arc-length parametrization of the kept positions: p(t) = P(s(t)).
% Then |dp/dt| = |P'(s)| * ds/dt = v, and integrating v returns s, which
% maps back to p through P. If the path length of v differs from the path
% length of p, v is rescaled by a global factor so that both end points match.
% Missing positions are replaced by the last available one (pf).
keep = iL:iR;
tk   = t(keep);
[pk, vk, scale] = speedConsistentPositions(tk, vOut(keep), pf(keep,:));
if isfinite(scale) && abs(scale - 1) > opts.maxScaleDev
    warning('filterBoundaryStrokes:pathLength', ...
        'Path lengths of v and p differ by %.1f%%; v was rescaled.', ...
        100*abs(scale - 1));
end

%% 8. Output assembly and consistency check
out = struct('p', pk, ...
             'v', vk, ...
             't', tk, ...
             'f', (numel(keep)-1) / (tk(end) - tk(1)));

% speedRelRmsErr:      rms(|dp/dt| - v) / max(v), central differences on t
% lengthRelErr:        |int v dt - polyline length of p| / int v dt
% maxShiftOutsideRamp: largest distance between derived and original
%                      positions on samples where v was not replaced
synth = synthetic(keep);
shift = vecnorm(pk - pf(keep,:), 2, 2);
speedP = vecnorm(centralDiff(pk, tk), 2, 2);
lenV   = trapz(tk, vk);
consistency = struct( ...
    'scale',               scale, ...
    'speedRelRmsErr',      sqrt(mean((speedP - vk).^2)) / max(vk), ...
    'lengthRelErr',        abs(lenV - sum(vecnorm(diff(pk), 2, 2))) / lenV, ...
    'maxShiftOutsideRamp', max([0; shift(~synth)]));

info = struct( ...
    'strokes',    table(b(1:end-1), b(2:end), dur, len, vmax, weak, ...
                        'VariableNames', {'iStart','iEnd','duration','length','vmax','belowThr'}), ...
    'lenThr',     lenThr, ...
    'noiseSpeed', noiseSpeed, ...
    'prominence', prom, ...
    'nCutLeft',   nL, ...
    'nCutRight',  nR, ...
    'idxLeft',    iL, ...
    'idxRight',   iR, ...
    'modeLeft',   modeL, ...
    'modeRight',  modeR, ...
    'synthetic',  synth, ...
    'keep',       keep(:), ...
    'consistency', consistency);
end

%% Local functions
function s = robustSigma(x)
% Noise sigma from the median absolute deviation.
s = 1.4826 * median(abs(x - median(x)));
end

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