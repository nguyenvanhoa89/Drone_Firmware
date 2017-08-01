%% Plot results
function Plot_Target_Estimated_Position (truth, est, uav)
    ntarget = size(truth.X,1);
    R_max = est.pf{1}.R_max;
%     k = max([est.foundIndex{:}]);
    k = size(uav(uav(3,:)>0),2);
    % c = get(gca,'ColorOrder');
    c = rand(ntarget+1,3);
    close(figure(1));
    hFig = figure(1);
    set(hFig, 'Position', [100 500 1000 1000]);
    % subplot(2,2,1);
    hold on;
    for i=1:ntarget
       htruth{i} = plot(truth.X{i}(1,1:est.foundIndex{i}-1),truth.X{i}(2,1:est.foundIndex{i}-1), 'Color' , c(i,:) , 'Marker' , '.','markersize',1);
    %    hest{i} = plot(est.X{i}(1,est.foundIndex{i}),est.X{i}(2,est.foundIndex{i}),'Color' , c(i,:) , 'Marker' , '*','markersize',10);
    end
    for i=1:ntarget
       htruth{i} = plot(truth.X{i}(1,est.foundIndex{i}),truth.X{i}(2,est.foundIndex{i}), 'LineWidth',2, 'Color' , c(i,:) , 'Marker' , 's','markersize',10,'MarkerFaceColor', c(i,:));
       hest{i} = plot(est.X{i}(1,est.foundIndex{i}),est.X{i}(2,est.foundIndex{i}),'LineWidth',2,'Color' , c(i,:) , 'Marker' , '*','markersize',10,'MarkerFaceColor', c(i,:));
    end
    huav = plot(uav(1,1:k), uav(2,1:k),'-', 'Color' , c(ntarget+1,:));
    % huav_real = plot(real_uav(1,1:k+Cycle+1), real_uav(2,1:k+Cycle+1),'-', 'Color' , 'r');
    hlegend = [];
    count = 0;
    for i=1:ntarget
        hlegend = [hlegend,htruth{i}, hest{i}];
        for j=1:2
            count = count + 1;
            if j ==1
                hLegendName{count} = ['Real trajectory of Target #', num2str(i)];
            else
                hLegendName{count} = ['Filtered of Target #', num2str(i)];
            end 
        end
    end
    hLegendName{count+1} = 'UAV Trajectory';
    % hLegendName{count+2} = 'Real UAV Trajectory';
    hlegend = [hlegend, huav];%,huav_real];
    legend(hlegend,hLegendName,'Location','best');
    hold off;
    grid on; 
    title('Position estimation with Particle filter & POMDP.', 'FontSize', 20);
    xlabel('x', 'FontSize', 20);
    ylabel('y', 'FontSize', 20);
    for i=1:ntarget
       labelpoints(est.X{i}(1,est.foundIndex{i}) +1,est.X{i}(2,est.foundIndex{i})+1, num2str(i)); 
    %    text(est.X{i}(1,est.foundIndex{i}) +1,est.X{i}(2,est.foundIndex{i})+1, num2str(i));
    end
    % axis([0,350,0,350]);
%     axis([0,R_max,0,R_max]);
    axis([-R_max,R_max,-R_max,R_max]);
end