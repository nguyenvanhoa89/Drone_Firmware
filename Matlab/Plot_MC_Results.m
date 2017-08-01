figure(2);
for mdp_cycle = 1:size(mdp_cycles,2)
    subplot(2,size(mdp_cycles,2),mdp_cycle);
    boxplot(cell2mat(MC_Results.RMS{mdp_cycle})'); % Generate box plot
    title(['RMS by Target, MDP = ',num2str(mdp_cycles(mdp_cycle)),' Cycle']);
    ylabel('RMS (m)');
    xlabel('Target #');
    grid on;
end
for mdp_cycle = 1:size(mdp_cycles,2)
    if mdp_cycle == 1
         MC_Results.MED_Set = mean(cell2mat(MC_Results.RMS{mdp_cycle}),1);
         MC_Results.Execution_Time_Set = MC_Results.Execution_Time{mdp_cycle};
         MC_Results.uav_travel_distance_set = MC_Results.uav_travel_distance{mdp_cycle};
    else
        MC_Results.MED_Set = [MC_Results.MED_Set;mean(cell2mat(MC_Results.RMS{mdp_cycle}),1)];
        MC_Results.Execution_Time_Set = [MC_Results.Execution_Time_Set;MC_Results.Execution_Time{mdp_cycle}];
        MC_Results.uav_travel_distance_set = [MC_Results.uav_travel_distance_set;MC_Results.uav_travel_distance{mdp_cycle}];
    end
end

subplot(2,size(mdp_cycles,2),size(mdp_cycles,2)+1);
boxplot(MC_Results.Execution_Time_Set',mdp_cycles); % Generate box plot
title('Execution Time by MDP');
ylabel('time (s)');
xlabel('MDP cycle');
grid on;
subplot(2,size(mdp_cycles,2),size(mdp_cycles,2)+2);
boxplot(MC_Results.uav_travel_distance_set',mdp_cycles); % Generate box plot
title('UAV Travel Distance by MDP');
ylabel('distance (m)');
xlabel('MDP cycle');
grid on;
subplot(2,size(mdp_cycles,2),size(mdp_cycles,2)+3);
boxplot(MC_Results.MED_Set',mdp_cycles); % Generate box plot
title('Mean Error Distance by MDP');
ylabel('time (s)');
xlabel('MDP cycle');
grid on;