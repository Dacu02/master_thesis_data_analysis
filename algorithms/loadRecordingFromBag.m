function traj = loadRecordingFromBag(bagFolder, topicName)
%ros2genmsg('../src/unisa_acg_ros2/haptics/haptic_experiments_msgs') % to put in main
    % read topicname from metadata.yaml file

    bagReader = ros2bagreader(bagFolder);
    bagSel = select(bagReader, "Topic", topicName);
    if bagSel.NumMessages == 0
        error('loadRecordingFromBag:empty', ...
            'Nessun messaggio sul topic "%s" in "%s".', topicName, bagFolder);
    end

    msgs = readMessages(bagSel);
    n = numel(msgs);

    pos = zeros(n, 3);
    linVel = zeros(n, 3);
    for i = 1:n
        m = msgs{i};
        %pos(i,:) = [m.position.x, m.position.z, m.position.y]; 
        pos(i,:) = [m.position.x, m.position.y, m.position.z];
        linVel(i,:) = [m.twist.linear.x, m.twist.linear.y, m.twist.linear.z];
    end

    t = bagSel.MessageList.Time;
    t = t - t(1);

    traj.p = pos;
    traj.v = sqrt(sum(linVel.^2, 2));
    traj.t = t;
    traj.f = 1 / median(diff(t));   % stimata dai timestamp reali, non fissa
    traj.m = struct('SourceFile', bagFolder, 'Topic', topicName);
end