function [out, info] = cutBoundaryStrokes(traj, opts)
%CUTBOUNDARYSTROKES  Remove low-intensity strokes at both ends of a
%   trajectory. Does not modify v or p inside the kept interval.
%
%   [out, info] = cutBoundaryStrokes(traj)
%   [out, info] = cutBoundaryStrokes(traj, Name, Value, ...)
%
%   Input traj (struct)
%       p  [N x 3]  position
%       v  [N x 1]  speed magnitude
%       t  [N x 1]  timestamps [s], strictly increasing
%       f  scalar   sampling frequency [Hz]
%
%   Output out (struct): same fields as traj, restricted to the kept
%       interval (N can decrease). v and p are unchanged inside it.
%   Output info (struct): strokes table, threshold, cut indices.
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
%       edgeFrac     ([0 0])  [left right] move each edge inward until the
%                             speed reaches edgeFrac*vmax of the edge stroke
%
%   No cap on the number of strokes removed per side: weak strokes are
%   removed inward until one stroke is not weak, always keeping at least
%   one stroke.

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
    opts.edgeFrac     (1,2) double {mustBeNonnegative} = [0 0]
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
% Strokes are removed from each side until one exceeds the threshold,
% always keeping at least one stroke.
nL = 0;
while nL < K-1 && weak(nL+1)
    nL = nL + 1;
end
nR = 0;
while nL + nR < K-1 && weak(K-nR)
    nR = nR + 1;
end
iL = b(nL+1);          % first kept sample
iR = b(K+1-nR);         % last kept sample

% Optional: move each edge inward along the flank of the edge stroke.
while opts.edgeFrac(1) > 0 && iL < iR && vs(iL) < opts.edgeFrac(1) * vmax(nL+1)
    iL = iL + 1;
end
while opts.edgeFrac(2) > 0 && iR > iL && vs(iR) < opts.edgeFrac(2) * vmax(K-nR)
    iR = iR - 1;
end

%% 6. Output assembly
keep = iL:iR;
out = struct('p', p(keep,:), ...
             'v', v(keep), ...
             't', t(keep), ...
             'f', (numel(keep)-1) / (t(iR) - t(iL)));

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
    'keep',       keep(:));
end

%% Local functions
function s = robustSigma(x)
% Noise sigma from the median absolute deviation.
s = 1.4826 * median(abs(x - median(x)));
end