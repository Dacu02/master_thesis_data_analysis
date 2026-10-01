function plotTrajectoriesN(trajs, labels, tabLabel)
    arguments
        trajs    (1,:) struct
        labels   (1,:) string = "sig" + string(1:numel(trajs))
        tabLabel (1,1) string = ""
    end

    nSig = numel(trajs);
    if numel(labels) ~= nSig
        error('plotTrajectoriesN:labels', 'labels must have %d elements, same as trajs.', nSig);
    end

    tabgroup = comparisonTabGroup();
    tab = uitab(tabgroup, 'Title', tabLabel);
    tl = tiledlayout(tab, 3, 1);

    sigColors = lines(nSig);
    axisColors = [1 0 0; 0 0.6 0; 0 0 1];
    axisNames = ["X", "Y", "Z"];
    lineStyles = ["-", "--", ":", "-."];

    ax1 = nexttile(tl);
    hold(ax1, 'on');
    for k = 1:nSig
        p = trajs(k).p;
        plot3(ax1, p(:,1), p(:,2), p(:,3), 'Color', sigColors(k,:), 'DisplayName', labels(k));
    end
    hold(ax1, 'off');
    axis(ax1, 'equal'); view(ax1, 3); grid(ax1, 'on');
    xlabel(ax1, 'X'); ylabel(ax1, 'Y'); zlabel(ax1, 'Z');
    legend(ax1, 'Location', 'best');
    title(ax1, 'Trajectory');

    ax2 = nexttile(tl);
    hold(ax2, 'on');
    for k = 1:nSig
        ls = lineStyles(mod(k-1, numel(lineStyles)) + 1);
        for c = 1:3
            plot(ax2, trajs(k).t, trajs(k).p(:,c), 'LineStyle', ls, 'Color', axisColors(c,:), ...
                'DisplayName', axisNames(c) + " " + labels(k));
        end
    end
    hold(ax2, 'off');
    grid(ax2, 'on');
    legend(ax2, 'Location', 'best', 'NumColumns', nSig);
    xlabel(ax2, 'Time (s)'); ylabel(ax2, 'Position');

    ax3 = nexttile(tl);
    hold(ax3, 'on');
    for k = 1:nSig
        plot(ax3, trajs(k).t, trajs(k).v, 'Color', sigColors(k,:), 'DisplayName', labels(k));
    end
    hold(ax3, 'off');
    grid(ax3, 'on');
    legend(ax3, 'Location', 'best');
    xlabel(ax3, 'Time (s)'); ylabel(ax3, 'Speed');

    linkaxes([ax2 ax3], 'x');

    tabgroup.SelectedTab = tab;
    drawnow limitrate
end