%% clear memory, screen, and close all figures
% To match with GPS coordinate in NE,UAV model in UAV_Control_Set,
% Get_Antenna_Gain or uav_model variable need to change from cos to sin and
% vice versa.
%
%
tic;
clear, clc, close all;

%% Process equation x[k] = sys(k, x[k-1], u[k]);
nx = 3;  % number of states
nuav = 4;
dt = 1; % second
q = 2;
ntarget = 10;
uav0 = [0;0;20;0];
sys = @(k, x, uk) x + uk; % random walk object
R_max = 250;
x0 = [R_max * (2*rand -1); R_max * (2*rand -1); 0];  
%% Initial variable
T = 900; % 15 minutes is max
% Time = 1;
pf.Ns = 3000;             % number of particles
Ms = 100; % 100 is current best
alpha = 0.5;
Area = [-R_max -R_max uav0(3);R_max R_max uav0(3)]';
theta_max = 5*pi/6; % max rotate angle (must less than pi) % current best 5*pi/6; pi/2 not work well
N_theta = 12; %12 is current best
Np = 1; % max velocity = Np * vu (m/s) % 2 with 5m/s is current best
vu = 10; % m/s
RSS_Threshold = -125; % dB
plot_box = 1;
%% PDF of process noise and noise generator function
% sigma_u = q^nx * eye(nx);
sigma_u = q^2 * [1 1 0];
Q = [];
nu = size(sigma_u,2); % size of the vector of process noise
p_sys_noise   = @(u) mvnpdf(u, zeros(1,nu), sigma_u);
gen_sys_noise = @(u) mvnrnd(zeros(1,nu),sigma_u,1)';         % sample from p_sys_noise (returns column vector)
%% PDF of observation noise and noise generator function
sigma_v = 5^2;
nv =  size(sigma_v,1);  % size of the vector of observation noise
p_obs_noise   = @(v) mvnpdf(v, zeros(1,nv), sigma_v);
gen_obs_noise = @(v) mvnrnd(zeros(1,nv),sigma_v,1)';         % sample from p_obs_noise (returns column vector)

%% Initial PDF
% p_x0 = @(x) normpdf(x, 0,sqrt(10));             % initial pdf
gen_x0 = @(x) [R_max* (2*rand -1) R_max* (2*rand -1) 0]';               % sample from p_x0 (returns column vector)

%% Observation likelihood PDF p(y[k] | x[k])
% (under the suposition of additive process noise)
p_yk_given_xk = @(k, yk, xk, uavk) p_obs_noise(yk - obs(k, xk, 0, uavk));

%% Separate memory space




%% Observation equation y[k] = obs(k, x[k], v[k]);
ht = 1;
hr = 50;
PtW = 0.5e-3;
Pt = 10*log10(PtW); %dBm
f = 146e6;
c = physconst('lightspeed');
lambda = c/f;
Gt = 0;   %dBm
Gr = -10; %dBm %Previous: -15
L = 10; %dBm 
d = @(x,uav) sqrt(sum((x-uav).^2)); % distance between UAV and target
ny = 1;                                           % number of observations
err_loc = 0.2;
phi = [2 3];
% gain_angle = load('3D_Directional_Gain_2Yagi_Element.txt'); % Theta	Phi	VdB	HdB	TdB
gain_angle = load('3D_Directional_Gain_Pattern.txt'); % Phi Theta	TdB
% obs = @(k, x, vk,uav) d(x,uav)     + vk ;     
% obs = @(k, x, vk,uav,gain_angle) friis(Pt, Gt, Gr, lambda, L, d(x,uav(1:3,:)),Get_Antenna_Gain(x, uav,gain_angle))     + vk ;     % (returns column vector)
obs = @(k, x, vk,uav,gain_angle) friis_2model(Pt, Gt, Gr, lambda, L, x,uav,Get_Antenna_Gain(x, uav,gain_angle))     + vk ;     % (returns column vector)
% [obsmin, obsmax] = Calculate_Mix_Max_Friss(Pt,Gt, Gr, L, R_max, uav0(3), err_loc, phi);

% obs = @(k, xk, vk) xk(1)^2/20 + vk;                  % (returns column vector)
% obs = @(k, x, vk) sin(x(1)) + vk ;                  % (returns column vector)


%% UAV Initialization
uav = zeros(nuav,T);
% uav(:,1) = uav0;
% uavtraj = uav_3d_circle([uav0; 0 ], T,dt, vu);
% uav = uavtraj(1:3,:);
uav(:,2) = uav0;
uav_model = @(x,dt,vu) [x(1) + dt*vu*sin(x(4)); % NE heading
            x(2) + dt*vu*cos(x(4)); 
            x(3)
            x(4) ];
fprintf('Filtering with PF...');
% fprintf('Press any key to see the result');
% pause 
%% Separate memory

pf.k               = 1;                   % initial iteration number
% pf.Ns              = 5000;                 % number of particles
pf.w               = ones(pf.Ns, T)/pf.Ns;     % weights
pf.particles       = [R_max * (2*rand(pf.Ns,1) -1) R_max * (2*rand(pf.Ns,1) -1) zeros(pf.Ns,1)]'; % particles
% pf.gen_x0          = gen_x0;              % function for sampling from initial pdf p_x0
pf.gen_x0          = [R_max * (2*rand(pf.Ns,1) -1) R_max * (2*rand(pf.Ns,1) -1) zeros(pf.Ns,1)]';
pf.p_yk_given_xk   = p_yk_given_xk;       % function of the observation likelihood PDF p(y[k] | x[k])
pf.gen_sys_noise   = gen_sys_noise;       % function for generating system noise
pf.gen_obs_noise   = gen_obs_noise;
pf.sigma_v         = sigma_v;
pf.sigma_u         = sigma_u;
pf.RSS_Threshold   = RSS_Threshold;
pf.R_max           = R_max;
pf.gain_angle      = gain_angle;
%pf.p_x0 = p_x0;                          % initial prior PDF p(x[0])
%pf.p_xk_given_ xkm1 = p_xk_given_xkm1;   % transition prior PDF p(x[k] | x[k-1])


%% Main Program

Time = 1;
% mdp_cycles = [1,3,5,7,10];
mdp_cycles = [5];
% MC_Results
for time = 1:Time
    fprintf('Iteration = %d/%d\n',time,Time);
    %% Generate true target state-space model
    if time == 1 %|| (1==1) % Initialize for 1st time. Change time to 1 (1 == 1) if want to get random target pos every run
        truth.X = cell(ntarget,1);
        for i=1:ntarget
            x(:,1) = [R_max * (2*rand -1); R_max * (2*rand -1); 0]; 
            truth.X{i} = x(:,1);
           for k=2:T
                x(:,k) = sys(k, x(:,k-1), gen_sys_noise()); 
                truth.X{i}= [truth.X{i} x(:,k)];
           end   
        end
    end

    for mdp_cycle = 1:size(mdp_cycles,2)
%         MC_Results.uav_travel_distance(time,mdp_cycle) = 0;
        est.foundTargetList = [];
        uav_travel_distance_k = 0;
        measurement = zeros(1,ntarget);
        Reward = zeros(1,ntarget);
%        disp(mdp_cycle) ;
        tic;
        %% Reset Estimation Initial
        est.X = cell(ntarget,1);
        est.pf = cell(ntarget,1);
        est.foundIndex = cell(ntarget,1);
        est.foundX = cell(ntarget,1);
        % Initial PF for each target estimation
        for i=1:ntarget 
           est.X{i} = zeros(nx,T);
           est.pf{i} = pf; 
           est.foundX{i} = zeros(nx,1);
           est.foundIndex{i} = 1;
        end
        %% Observation Initial
        meas.Z = cell(ntarget,1);
        meas.ValidZCount = cell(ntarget,1);
        meas.Reward = cell(ntarget,1);
        meas.UAV = cell(ntarget,1);
        % Initial measurement for each target estimation
        for i=1:ntarget 
            meas.Z{i} = zeros(1,T);
            meas.ValidZCount{i} = 0;
            meas.UAV{i} = zeros(nuav,T);
            meas.Reward{i} = zeros(1,T);
        end
        % Main program for estimation
        for k = 2:T
            fprintf('Iteration = %d/%d\n',k,T);
            [UAV_Sets,Cost_Sets] = UAV_Control_Sets(uav(:,k),Np,N_theta,dt,vu,theta_max);

            for i=1:ntarget 
               measurement(i) = obs(k, truth.X{i}(:,k),   gen_obs_noise(),uav(:,k),pf.gain_angle);
               if measurement(i) < RSS_Threshold || ~ isempty(est.foundTargetList(i == est.foundTargetList))
                   measurement(i) = RSS_Threshold;
               end
            end
            if mean(measurement) == RSS_Threshold
                TargetList = 1:1:ntarget;
                TargetList ( est.foundTargetList) = [];
                best_u = TargetList(randi(size(TargetList,2)));
            else
                [~,best_u] = max(measurement,[],2);
            end

            for i=1:ntarget 
               if isempty(est.foundTargetList(i == est.foundTargetList))
                   est.pf{i}.k = k;
                   if measurement(i) > RSS_Threshold
                       meas.ValidZCount{i} = meas.ValidZCount{i} + 1;
                       meas.Z{i}(k)  = measurement(i);
                       [est.X{i}(:,k), est.pf{i}] = bootstrap_filter (k, est.pf{i}, sys, obs, meas.Z{i}(k), uav(:,k));               
                   else
                       est.X{i}(:,k) = sys(k, est.X{i}(:,k-1), gen_sys_noise());
                       est.pf{i}.w(:,k) =  est.pf{i}.w(:,k-1);
                       est.pf{i}.particles(:,:,k) = est.pf{i}.particles(:,:,k-1);
                   end
                   if i == best_u && mod(k,mdp_cycles(mdp_cycle))==0 % Update way point every 5 seconds only
                        [meas.UAV{i}(:,k),meas.Reward{i}(k)]  = Control_Vector_Selection (k, est.pf{i},sys,obs, UAV_Sets, Ms, alpha, Area);
                   
                   else
                       u =  uav_model(uav(:,k),dt,vu);
                       if ~isempty(find(sign(u(1:3) - Area(:,1))==-1)) || ~isempty(find(sign(Area(:,2) - u(1:3)) ==-1)) % check if out of predefine area
                           [meas.UAV{i}(:,k),meas.Reward{i}(k)]  = Control_Vector_Selection (k, est.pf{i},sys,obs, UAV_Sets, Ms, alpha, Area);
                       else
                           meas.Reward{i}(k) = -1; % Min value to skip
                           meas.UAV{i}(:,k) = u;
                       end
                   end
                   % Terminate condition
                   if meas.ValidZCount{i}>5 && det(cov(est.pf{i}.particles(1:2,:,k)')) < pf.Ns 
                       est.foundX{i} = est.X{i}(:,k);
                       est.foundIndex{i} = k;
                       est.foundTargetList = [est.foundTargetList i];
                   end
               else
                   meas.Reward{i}(k) = -1; % Min value to skip
               end
               Reward(i) = meas.Reward{i}(k); %% Dummy variable to store current reward function
               if Reward(i) == -1e30 % If all of way points out of pre-defined area, back to previous position
                   meas.UAV{i}(:,k) = uav(:,k-1);
               end
            end
        %     [~,best_u] = max(Reward,[],2);
            uav(:,k+1) = meas.UAV{best_u}(:,k);
            uav_travel_distance_k = uav_travel_distance_k + norm(uav(:,k+1) - uav(:,k));
            if size(est.foundTargetList,2) == ntarget || (norm(uav(:,k))/(dt*vu) + k) > T
               break; 
            end
        end
        for i=1:ntarget
            if isempty(est.foundTargetList(i == est.foundTargetList))
               est.foundX{i} = est.X{i}(:,k);
               est.foundIndex{i} = k;
            end
            est.RMSFound{i} = d( est.foundX{i}, truth.X{i}(:,est.foundIndex{i})) ;
        end
        MC_Results.uav_travel_distance{mdp_cycle}(:,time)= uav_travel_distance_k;
        MC_Results.Execution_Time{mdp_cycle}(:,time) = toc;
        MC_Results.MED{mdp_cycle}(:,time)= mean(cell2mat(est.RMSFound));
        MC_Results.RMS{mdp_cycle}(:,time)= est.RMSFound;
        MC_Results.foundIndex{mdp_cycle}(:,time) = est.foundIndex;
%         if max(cell2mat(est.foundIndex),[],1) > 500
%            disp(max(cell2mat(est.foundIndex),[],1));
%         end
    end 
end
%% Report result
clear MC_Results_Summary;
for mdp_cycle = 1:size(mdp_cycles,2)
   MC_Results_Summary.uav_travel_distance(:,mdp_cycle) = [median(MC_Results.uav_travel_distance{mdp_cycle});mean(MC_Results.uav_travel_distance{mdp_cycle}); std(MC_Results.uav_travel_distance{mdp_cycle})];
   MC_Results_Summary.Execution_Time(:,mdp_cycle)= [median(MC_Results.Execution_Time{mdp_cycle}); mean(MC_Results.Execution_Time{mdp_cycle}); std(MC_Results.Execution_Time{mdp_cycle})];
   MC_Results_Summary.RMS{mdp_cycle} = [ median(cell2mat(MC_Results.RMS{mdp_cycle}),2)';mean(cell2mat(MC_Results.RMS{mdp_cycle}),2)';std(cell2mat(MC_Results.RMS{mdp_cycle}),0,2)'];
   MC_Results_Summary.foundIndex{mdp_cycle} = [median(cell2mat(MC_Results.foundIndex{mdp_cycle}),2)';mean(cell2mat(MC_Results.foundIndex{mdp_cycle}),2)'; std(cell2mat(MC_Results.foundIndex{mdp_cycle}),0,2)'];
end
    
for i=1:ntarget
    if isempty(est.foundTargetList(i == est.foundTargetList))
       est.foundX{i} = est.X{i}(:,k);
       est.foundIndex{i} = k;
    end
end
for i=1:ntarget 
   est.RMS{i} = d( est.X{i}, truth.X{i}) ;
end
for i=1:ntarget 
   est.RMSFound{i} = d( est.foundX{i}, truth.X{i}(:,est.foundIndex{i})) ;
end
%% Back to station
pstart = uav(:,k);
pend = uav0;
v1 = pend(1:2,:) - pstart(1:2,:) ;
v2 = [cos(pstart(4,:)); sin(pstart(4,:))];
% v2 = theta;
x1 = v1(1,:); y1 = v1(2,:);
x2 = v2(1,:); y2 = v2(2,:);
theta=atan2(x1.*y2-y1.*x2,dot(v1,v2));
Cycle = ceil(norm(pstart(1:2) - pend(1:2)) / (dt * vu)) + ceil(abs(theta/theta_max));
uav(:,k+1:k+Cycle+1) = uav_backto_station(pstart, pend,theta_max, dt,vu);
est.foundIndex
est.RMSFound
uav_travel_distance_k = uav_travel_distance_k + norm(pstart(1:2) - pend(1:2))
time_PF = toc;
fprintf('PF Execution Time         :%5.2fs\n',time_PF);
% 
% %% Estimate state
% for time = 1:Time
%     uav_travel_distance(time,k) = 0;
%     uav_travel_distance_k(time) = 0;
%     for k = 2:T
%         
%        fprintf('Iteration = %d/%d\n',k,T);
%        % state estimation
%        pf.k = k;
%        % Take measurement
%        y(:,k) = obs(k, x(:,k),   v(:,k),uav(:,k));
% %        [xh(:,k), pf] = particle_filter(sys,obs,uav(:,k), y(:,k), pf, 'SIR');
%        [xh(:,k), pf] = bootstrap_filter (k, pf, sys, obs, y(:,k), uav(:,k));
%        %    [xh(:,k), pf] = particle_filter(sys, y(:,k), pf, 'systematic_resampling');   
% 
%        % filtered observation
%        yh(:,k) = obs(k, xh(:,k), 0,uav(:,k));
%     %    Reward_From_Actions(U_Sets(:,3),pf,obs,200,5,0.5,xh,gen_obs_noise);
%         % Calculate next UAV position
%        [UAV_Sets,Cost_Sets] = UAV_Control_Sets(uav(:,k),Np,N_theta,dt,vu,theta_max);
% %         Reward = Reward_From_Actions(UAV_Sets,pf,sys,obs,Ms,k,alpha,xh,[0 0 uav0(3);R_max R_max uav0(3)]');% + Cost_Sets * 2e-4;
% %         [~,best_u] = max(Reward,[],2);
% %         uav(:,k+1) = U_Sets(:,best_u);
%        uav(:,k+1) = Control_Vector_Selection (k, pf,sys,obs, UAV_Sets, Ms, alpha, Area);
%        uav_travel_distance_k(time) = uav_travel_distance_k(time) + norm(uav(:,k+1) - uav(:,k));
%        uav_travel_distance(time,k) = uav_travel_distance_k(time);
% 
%        if(mod(k,1)==0) %T or 1
%             clf;
%             hFig = figure(1);
%             set(hFig, 'Position', [100 500 1000 350]);
%             subplot(1,2,1);
%             hold on;
%             plot(x(1,1:k),x(2,1:k),'g+','markersize',10);
%             plot(xh(1,1:k),xh(2,1:k),'b.');
%             plot(uav(1,1:k), uav(2,1:k), 'r-');
%             hold off;
%             grid on;
%             xlabel('x');
%             ylabel('y');
%             axis([0,R_max,0,R_max]);
%             legend('Real trajectory', 'Filtered','UAV Trajectory','Location','best');
%             title('Position estimation with Particle filter.');
% 
%             subplot(1,2,2);
%             ptk = pf.particles(:,:,k);
%             pwk = pf.w(:,k);
%             scatter(ptk(1,:),ptk(2,:),[],pwk,'filled');
%             title('Particle Distribution');
%             grid on;
%             axis([0,R_max,0,R_max]);
%             xlabel('x');
%             ylabel('y');
%             colorbar;
%             drawnow;     
% 
%        end
%        %}
%        RMS_Distance(time,k) = d(xh(:,k), x(:,k)) ;
%        RMS_Distance_k(time) = RMS_Distance(time,k);
%        if d(xh(:,k), x(:,k)) < 10^(sigma_v/20)
% %           break; 
%        end
%        if norm(std(pf.particles(:,:,k),0,2)) < sigma_v/3 %&& k>5 &&  abs(mean(y(k-5:k))- mean(yh(k-5:k))) < sigma_v/5 
%            break;
%        end
% %        if k>5 &&  mean(y(k-5:k))- mean(yh(k-5:k)) > sigma_v/10 && std(yh(k-5:k)) < sigma_v/10
% %           break; 
% %        end
%     % pause(1);
%     end
%     time = time + 1;
% end
% 
% 
%% Plot results
ntarget = size(truth.X,1);
R_max = est.pf{1}.R_max;
k = max([est.foundIndex{:}]);
% c = get(gca,'ColorOrder');
c = rand(ntarget+1,3);
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
huav = plot(uav(1,1:k+Cycle+1), uav(2,1:k+Cycle+1),'-', 'Color' , c(ntarget+1,:));
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
hlegend = [hlegend, huav];
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
axis([0,350,0,350]);
axis([-R_max,R_max,-R_max,R_max]);
%{
iptsetpref('ImshowBorder','tight');
set(hFig,'Color','white');
print(hFig,'-depsc2','-painters','SIM_Estimate_Postion.eps');
 !ps2pdf -dEPSCrop SIM_Estimate_Postion.eps
%}

save(['MultiTarget_Estimate',datestr(now, 'yyyymmddHHMMss'),'L5DB_5Cycles.mat'], 'truth', 'meas', 'est','uav_travel_distance_k','uav','MC_Results','mdp_cycles' );
% subplot(2,2,2);
% hold on;
% plot(x(1,k),x(2,k),'+g','markersize',15);
% plot(xh(1,k),xh(2,k),'.b','markersize',15);
% legend('Real trajectory', 'Filtered','Location','best');
% title('Final Position estimation with Particle filter.');
% xlabel('x');
% ylabel('y');
% grid on;
% axis([0,R_max,0,R_max]);
% hold off;
% ptk = pf.particles(:,:,k);
% pwk = pf.w(:,k);
% 
% subplot(2,2,3);
% scatter(ptk(1,:),ptk(2,:),[],pwk,'filled');title('Particle Distribution');
% grid on;
% colorbar;
% xlabel('x');
% ylabel('y');
% axis([0,R_max,0,R_max]);
% 
% subplot(2,2,4);
% plot(1:k, RMS_Distance(Time,1:k),'-k');
% title('RMS Trend by time');
% legend('RMS');
% grid on;
% xlabel('cycle');
% ylabel('RMS (m)');
% 
% RMS =  d(xh(:,k), x(:,k));
% pose_diff = xh(:,k)- x(:,k)
% RSS_diff = y(k) - yh(k)
% fprintf('RMS of PF                 :%5.2fm\n',RMS);
% time_PF = toc;
% fprintf('PF Execution Time         :%5.2fs\n',time_PF);
% % fprintf('UAV Travel Distance       :%5.2f\n',uav_travel_distance);
% % 
% if Time > 1
%     s1 = RMS_Distance(:,15); s2 = RMS_Distance(:,25); 
%     % s3 = RMS_Distance(:,35); s4 = RMS_Distance(:,50); 
%     figure(2)
%     % boxplot([s1 s2 s3 s4],'notch','on','labels',{'k=15','k=25','k=35','k=50'});
%     boxplot([s1 s2],'notch','on','labels',{'k=15','k=25'});
%     xlabel('cycle');
%     ylabel('RMS (m)');
%     title('RSM error by cycle');
%     grid on;
% 
%     figure(3)
%     u_travel_1 = uav_travel_distance(:,15); u_travel_2 = uav_travel_distance(:,25);
%     % u_travel_3 = uav_travel_distance(:,35); u_travel_4 = uav_travel_distance(:,50);
%     % boxplot([u_travel_1 u_travel_2 u_travel_3 u_travel_4],'notch','on','labels',{'k=15','k=25','k=35','k=50'});
%     boxplot([u_travel_1 u_travel_2 ],'notch','on','labels',{'k=15','k=25'});
%     grid on;
%     xlabel('cycle');
%     ylabel('distance (m)');
%     title('UAV travel distance by cycle');
%     figure(4)
%     boxplot(RMS_Distance_k,'notch','on');
%     title('RSM error by cycle');
%     ylabel('distance (m)');
%     
%     figure(5)
%     boxplot(uav_travel_distance_k,'notch','on');
%     title('UAV travel distance by cycle');
%     ylabel('distance (m)');
%     save(['Distance',datestr(now, 'yyyymmddHHMMss'),'.mat'], 'RMS_Distance', 'uav_travel_distance','pf','Ms','N_theta','Np','vu','R_max','time_PF');
%     save(['Distance_k',datestr(now, 'yyyymmddHHMMss'),'.mat'], 'RMS_Distance_k', 'uav_travel_distance_k','pf','Ms','N_theta','Np','vu','R_max','time_PF');
% 
% end
return;