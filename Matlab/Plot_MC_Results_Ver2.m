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

h = figure(2);
subplot(1,3,1);
boxplot(MC_Results.MED_Set',mdp_cycles); % Generate box plot
title('Mean Error Distance by POMDP', 'FontSize', 20);
ylabel('Error (m)', 'FontSize', 20);
xlabel('POMDP cycle', 'FontSize', 20);
grid on;
subplot(1,3,2);
boxplot(MC_Results.Execution_Time_Set',mdp_cycles); % Generate box plot
title('Execution Time by POMDP', 'FontSize', 20);
ylabel('Time (s)', 'FontSize', 20);
xlabel('POMDP cycle', 'FontSize', 20);
grid on;
subplot(1,3,3);
boxplot(MC_Results.uav_travel_distance_set',mdp_cycles); % Generate box plot
title('UAV Travel Distance by POMDP', 'FontSize', 20);
ylabel('Distance (m)', 'FontSize', 20);
xlabel('POMDP cycle', 'FontSize', 20);
grid on;
%{
iptsetpref('ImshowBorder','tight');
set(h,'Color','white');
print(h,'-depsc2','-painters','SIM_POMDP_Cycles.eps');
 !ps2pdf -dEPSCrop SIM_POMDP_Cycles.eps
%}
