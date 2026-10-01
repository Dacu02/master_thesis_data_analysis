function tabgroup = comparisonTabGroup()
% utility to share a single tabgroup across multiple calls to computeMetrics 
% and avoid idelog bad management of plots
    persistent fig tg
    if isempty(fig) || ~isvalid(fig)
        fig = figure('Name', 'Confronti', 'NumberTitle', 'off', ...
            'MenuBar', 'figure', 'ToolBar', 'figure', ...
            'HandleVisibility', 'off');
        tg = uitabgroup(fig);
    end
    tabgroup = tg;
end