function metrics = computeMetrics(reference, test, label, t)
    arguments
        reference (:,:) double
        test (:,:) double
        label (1,1) string = ""
        t (:,1) double = double.empty(0,1)
    end

    if ~isequal(size(reference), size(test))
        error('computeMetrics:size', ...
            'reference e test devono avere la stessa dimensione (%s vs %s).', ...
            mat2str(size(reference)), mat2str(size(test)));
    end
    if ~isempty(t) && numel(t) ~= size(reference, 1)
        error('computeMetrics:time', ...
            't ha %d elementi, attesi %d (come reference/test).', numel(t), size(reference, 1));
    end

    err = reference - test;
    perSampleErr = vecnorm(err, 2, 2);

    validRows = ~any(isnan([reference, test]), 2);
    nInvalid = nnz(~validRows);
    if nInvalid > 0
        warning('computeMetrics:nan', ...
            '%d/%d campioni con NaN esclusi dal calcolo di SNR/MSE (restano visibili nel plot come gap).', ...
            nInvalid, numel(validRows));
    end

    refValid = reference(validRows, :);
    errValid = err(validRows, :);
    metrics.SNRdB = 10*log10(sum(refValid(:).^2) / sum(errValid(:).^2));
    metrics.MSE = mean(perSampleErr(validRows).^2);

    if label ~= ""
        plotComparisonTab(reference, test, err, perSampleErr, metrics, label, t);
    end
end