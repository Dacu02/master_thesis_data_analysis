function metrics = computeMetrics(reference, test, label)
%COMPUTEMETRICS Metriche di qualità tra due segnali (es. due velocità).
%   metrics = COMPUTEMETRICS(reference, test) restituisce SNR (dB) e
%   MSE. Se label non è vuota, disegna anche un confronto non
%   bloccante con quella stringa come titolo della scheda: ogni
%   chiamata con una label aggiunge una scheda a un'unica figura
%   condivisa, così più chiamate nello stesso loop si impilano nelle
%   schede invece di aprire finestre separate. Senza label (default),
%   nessun plot.
    arguments
        reference (:,1) double
        test (:,1) double
        label (1,1) string = ""
    end

    if numel(reference) ~= numel(test)
        error('computeMetrics:size', 'reference e test devono avere la stessa lunghezza.');
    end

    err = reference - test;
    metrics.SNRdB = 10*log10(sum(reference.^2) / sum(err.^2));
    metrics.MSE = mean(err.^2);

    if label ~= ""
        plotComparisonTab(reference, test, err, metrics, label);
    end
end

function plotComparisonTab(reference, test, err, metrics, label)
    persistent fig tabgroup
    if isempty(fig) || ~isvalid(fig)
        fig = figure('Name', 'computeMetrics — confronti', 'NumberTitle', 'off', ...
            'MenuBar', 'figure', 'ToolBar', 'figure');
        tabgroup = uitabgroup(fig);
    end

    tab = uitab(tabgroup, 'Title', label);
    tl = tiledlayout(tab, 2, 1);

    ax1 = nexttile(tl);
    plot(ax1, reference, 'DisplayName', 'reference'); hold(ax1, 'on');
    plot(ax1, test, 'DisplayName', 'test'); hold(ax1, 'off');
    legend(ax1, 'Location', 'best');
    title(ax1, sprintf('SNR = %.2f dB, MSE $=%.2f$', metrics.SNRdB, metrics.MSE));
    ylabel(ax1, 'Ampiezza');

    ax2 = nexttile(tl);
    plot(ax2, err);
    xlabel(ax2, 'Campione'); ylabel(ax2, 'Errore (reference - test)');

    tabgroup.SelectedTab = tab;
    drawnow limitrate
end