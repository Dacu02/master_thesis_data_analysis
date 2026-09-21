function trajectory = filterStrokes(trajectory, strokes, D_threshold)
% FILTERSTROKES Filters strokes based on a threshold on D
%   trajectory = filterStrokes(trajectory, strokes, D_threshold) filters the input strokes based on the provided D_threshold. It returns a trajectory structure containing the filtered strokes.
%   INPUT
%     trajectory  : structure containing the original trajectory data (time, position, velocity) 
%     strokes     : structure containing 'D', 'Mu', 'Sigma', 'to' fields for each stroke
%     D_threshold : threshold value for filtering strokes based on their 'D' field
%   OUTPUT
%     trajectory  : structure containing the filtered strokes and their corresponding time, position, and velocity

    filtered_strokes = [];
    for i = 1:length(strokes)
        if strokes(i).D >= D_threshold
            trim_up_to = i;
        end
    end
    for i = length(strokes):-1:1
        if strokes(i).D >= D_threshold
            trim_from = i;
            break;
        end
    end

    left_strokes = strokes(trim_up_to:trim_from);
    trajectory.strokes = filtered_strokes;


end