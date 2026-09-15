function merged = mergeStructs(defaults, override)
    % Overrides struct fields with the one passed in override
    merged = defaults;
    if isempty(override)
        return
    end
    fields = fieldnames(override);
    for i = 1:numel(fields)
        f = fields{i};
        if isfield(merged, f) && isstruct(merged.(f)) && isstruct(override.(f)) ...
                && isscalar(merged.(f)) && isscalar(override.(f))
            merged.(f) = mergeStructs(merged.(f), override.(f));
        else
            merged.(f) = override.(f);
        end
    end
end