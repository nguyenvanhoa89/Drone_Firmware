%% Plot results
function Plot_Truth_UAV (truth, uav, c)
    ntarget = size(truth.X,1);
    R_max = 500;
%     k = max([est.foundIndex{:}]);
    k = size(uav(uav(3,:)>0),2)+1;
    % c = get(gca,'ColorOrder');
%     c = rand(ntarget+1,3);
%     close(figure(1));
    hFig = figure(1);
    set(hFig, 'Position', [0 0 800 550]);
    % subplot(2,2,1);
    hold on;
    for i=1:ntarget
       htruth{i} = plot(truth.X{i}(1,2:k),truth.X{i}(2,2:k), 'LineWidth',1, 'Color' , c(i,:) , 'Marker' , '.','markersize',1);
    end
%                 huav = plot(uav(1,k), uav(2,k),'-', 'Color' , c(ntarget+1,:),'LineWidth',3);

    huav = plot(uav(1,2:k), uav(2,2:k),'-', 'Color' , c(ntarget+1,:),'LineWidth',3);
    % huav_real = plot(real_uav(1,1:k+Cycle+1), real_uav(2,1:k+Cycle+1),'-', 'Color' , 'r');
    hlegend = [];
    count = 0;
    for i=1:1
        hlegend = [hlegend,htruth{i}];
        for j=1:1
            count = count + 1;
            if j ==1
                hLegendName{count} = ['Real trajectory of Target'];
            else
                hLegendName{count} = ['Estimated location of Target'];
            end 
        end
    end
    hLegendName{count+1} = 'UAV Trajectory';
    % hLegendName{count+2} = 'Real UAV Trajectory';
    hlegend = [hlegend, huav];%,huav_real];
    legend(hlegend,hLegendName,'Location','best');
%                 hold off;
    grid on; 
    title('Position estimation with Particle filter & POMDP in SITL.', 'FontSize', 10);
    xlabel('East (m)', 'FontSize', 10);
    ylabel('North (m)', 'FontSize', 10);
    if k ==2 
        for i=1:ntarget
           labelpoints(truth.X{i}(1,k) +1,truth.X{i}(2,k)+1, num2str(i), 'FontSize', 10); 
        %    text(est.X{i}(1,est.foundIndex{i}) +1,est.X{i}(2,est.foundIndex{i})+1, num2str(i));
        end
    end

    set(gca, 'FontSize', 10);
    % axis([0,350,0,350]);
    axis([-10,R_max,-10,R_max]);
end