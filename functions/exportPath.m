function output_path = exportPath(input_path, database_type)
    current = pwd;
    assert(startsWith(input_path, [current filesep]), ...
           'Path is not in the current directory.');
    relative_part = input_path(length(current)+2 : end);
    path_parts = strsplit(relative_part, filesep);

    if database_type == "PATH"
        if length(path_parts) ~= 5
            error("Path %s is not valid for database type 'PATH'.", input_path);
        end
        subject_folder = path_parts{2};
        task_folder = path_parts{3};
        recording_folder = path_parts{4};

        recording_number = regexp(recording_folder, 'recording_(\d+)', 'tokens', 'once');
        if isempty(recording_number)
            error("Path not valid for database type 'PATH'.");
        end
        recording_number = recording_number{1};
        output_path = fullfile(subject_folder, task_folder, strcat('recording_', recording_number));
    else
        if database_type == "FORCE"
            error("Not implemented yet: exportPath for database type 'FORCE'.");
        end
    end
end