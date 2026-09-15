function metrics = compareTrajectories(trajReference, trajTest, label)
    arguments
        trajReference (1,1) struct
        trajTest (1,1) struct
        label (1,1) string = ""
    end

    posMetrics = computeMetrics(trajReference.p, trajTest.p);
    velMetrics = computeMetrics(trajReference.v, trajTest.v);

    metrics.SNR_T = posMetrics.SNRdB;
    metrics.SNR_V = velMetrics.SNRdB;

    if label ~= ""
        plotTrajectoryComparisonTab(trajReference, trajTest, metrics, label);
    end
end

function plotTrajectoryComparisonTab(trajReference, trajTest, metrics, label)
    tabgroup = comparisonTabGroup();
    tab = uitab(tabgroup, 'Title', label);
    tl = tiledlayout(tab, 3, 1);

    ref = trajReference.p;
    test = trajTest.p;
    t = trajReference.t;   % == trajTest.t dopo l'allineamento in main

    ax1 = nexttile(tl);
    plot3(ax1, ref(:,1), ref(:,2), ref(:,3), 'DisplayName', 'reference'); hold(ax1, 'on');
    plot3(ax1, test(:,1), test(:,2), test(:,3), 'DisplayName', 'test'); hold(ax1, 'off');
    axis(ax1, 'equal'); view(ax1, 3); grid(ax1, 'on');
    xlabel(ax1, 'X'); ylabel(ax1, 'Y'); zlabel(ax1, 'Z');
    legend(ax1, 'Location', 'best');
    title(ax1, sprintf('Traiettoria — SNR_T = %.2f dB', metrics.SNR_T));

    ax2 = nexttile(tl);
    colors = [1 0 0; 0 0.6 0; 0 0 1];
    names = ["X", "Y", "Z"];
    hold(ax2, 'on');
    for k = 1:3
        plot(ax2, t, ref(:,k), '-', 'Color', colors(k,:), 'DisplayName', names(k) + " ref");
        plot(ax2, t, test(:,k), '--', 'Color', colors(k,:), 'DisplayName', names(k) + " test");
    end
    hold(ax2, 'off');
    grid(ax2, 'on');
    legend(ax2, 'Location', 'best', 'NumColumns', 3);
    xlabel(ax2, 'Tempo (s)'); ylabel(ax2, 'Posizione');

    ax3 = nexttile(tl);
    plot(ax3, t, trajReference.v, 'DisplayName', 'reference'); hold(ax3, 'on');
    plot(ax3, t, trajTest.v, 'DisplayName', 'test'); hold(ax3, 'off');
    grid(ax3, 'on');
    legend(ax3, 'Location', 'best');
    title(ax3, sprintf('Velocità — SNR_V = %.2f dB', metrics.SNR_V));
    xlabel(ax3, 'Tempo (s)'); ylabel(ax3, 'Velocità');

    tabgroup.SelectedTab = tab;
    drawnow limitrate
end