%% clear memory, screen, and close all figures
tic;
clear, clc, close all;

%% Process equation x[k] = sys(k, x[k-1], u[k]);
nx = 3;  % number of states
nuav = 4;
dt = 1; % second
q = 2;
ntarget = 5;
uav0 = [0;0;20;0];
sys = @(k, x, uk) x + uk; % random walk object
R_max = 500;
x0 = [R_max * rand; R_max * rand; 0];  
%% Initial variable
T = 900; % 15 minutes is max
% Time = 1;
pf.Ns = 3000;             % number of particles
Ms = 100; % 100 is current best
alpha = 0.5;
Area = [0 0 uav0(3);R_max R_max uav0(3)]';
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
gen_x0 = @(x) [R_max* rand R_max* rand 0]';               % sample from p_x0 (returns column vector)

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
uav_model = @(x,dt,vu) [x(1) + dt*vu*cos(x(4)); 
            x(2) + dt*vu*sin(x(4)); 
            x(3)
            x(4) ];
fprintf('Filtering with PF...');
% fprintf('Press any key to see the result');
% pause 
%% Separate memory

pf.k               = 1;                   % initial iteration number
% pf.Ns              = 5000;                 % number of particles
pf.w               = ones(pf.Ns, T)/pf.Ns;     % weights
pf.particles       = [R_max * rand(pf.Ns,1) R_max * rand(pf.Ns,1) zeros(pf.Ns,1)]'; % particles
% pf.gen_x0          = gen_x0;              % function for sampling from initial pdf p_x0
pf.gen_x0          = [R_max * rand(pf.Ns,1) R_max * rand(pf.Ns,1) zeros(pf.Ns,1)]';
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
            x(:,1) = [R_max * rand; R_max * rand; 0]; 
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
%             meas.Reward{i} = zeros(size(UAV_Sets,2),T);
        end
        % Main program for estimation
        for k = 2:T
            fprintf('Iteration = %d/%d\n',k,T);
            bestTarget = 0;
            [UAV_Sets,Cost_Sets] = UAV_Control_Sets(uav(:,k),Np,N_theta,dt,vu,theta_max);
            meas.Reward{i}(:,k) = zeros(1,size(UAV_Sets,2))';
            for i=1:ntarget 
               measurement(i) = obs(k, truth.X{i}(:,k),   gen_obs_noise(),uav(:,k),pf.gain_angle);
               if measurement(i) <= RSS_Threshold %|| ~ isempty(est.foundTargetList(i == est.foundTargetList))
                   measurement(i) = RSS_Threshold;
               end
            end
            if mean(measurement) == RSS_Threshold
                TargetList = 1:1:ntarget;
                TargetList ( est.foundTargetList) = [];
                bestTarget = TargetList(randi(size(TargetList,2)));
            end
            for i=1:ntarget 
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
            end
            if bestTarget > 0
                [uav(:,k+1),~]  = Control_Vector_Selection (k, est.pf{bestTarget},sys,obs, UAV_Sets, Ms, alpha, Area);   
            else
               for i =1:ntarget
                   if measurement(i) > RSS_Threshold && mod(k,mdp_cycles(mdp_cycle))==0 % Update way point every 5 seconds only
                        [meas.Reward{i}(:,k)]  = Control_Vector_Selection_For_Tracking (k, est.pf{i},sys,obs, UAV_Sets, Ms, alpha, Area);
                   else
                       u =  uav_model(uav(:,k),dt,vu);
                       if ~isempty(find(sign(u(1:3) - Area(:,1))==-1)) || ~isempty(find(sign(Area(:,2) - u(1:3)) ==-1)) % check if out of predefine area
                           [meas.Reward{i}(:,k)]  = Control_Vector_Selection_For_Tracking (k, est.pf{i},sys,obs, UAV_Sets, Ms, alpha, Area);
                       else
                           meas.Reward{i}(:,k) = 0* ones(1,size(UAV_Sets,2))'; % Min value to skip
                       end
                   end
                   % Terminate condition
                   if meas.ValidZCount{i}>5 && det(cov(est.pf{i}.particles(1:2,:,k)')) < pf.Ns && ~ismember(i,est.foundTargetList)
                       est.foundX{i} = est.X{i}(:,k);
                       est.foundIndex{i} = k;
                       est.foundTargetList = [est.foundTargetList i];
                   end 
                end
                for i =1:ntarget
                    if ismember(i,est.foundTargetList)
                       rewardWeight = 1/(size(est.foundTargetList,2)+1); 
                    else
                       rewardWeight = 1;
                    end
                    if i==1
                        meas.Reward_Set =  rewardWeight* meas.Reward{i}(:,k)*measurement(i)/RSS_Threshold;
                    else
                        meas.Reward_Set  = [meas.Reward_Set ,rewardWeight* meas.Reward{i}(:,k)*measurement(i)/RSS_Threshold];
                    end
                end
                if sum(sum(meas.Reward_Set)) == 0
                    uav(:,k+1)= uav_model(uav(:,k),dt,vu);
                else
                    [~,best_u] = max(sum(meas.Reward_Set,2),[],1);
                    uav(:,k+1) = UAV_Sets(:,best_u);
                end
            end
            uav_travel_distance_k = uav_travel_distance_k + norm(uav(:,k+1) - uav(:,k));
            if (norm(uav(:,k))/(dt*vu) + k) > T || size(est.foundTargetList,2) == ntarget
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
%% Plot results
ntarget = size(truth.X,1);
R_max = est.pf{1}.R_max;
% k = max([est.foundIndex{:}]);
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
huav = plot(uav(1,1:k+Cycle+1), uav(2,1:k+Cycle+1),'-', 'Color' , c(ntarget+1,:),'LineWidth',2);
% plot(uav(1,k+Cycle+1:863), uav(2,k+Cycle+1:863),'-', 'Color' , rand(1,3),'LineWidth',2);
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
title('Position tracking with Particle filter & POMDP.', 'FontSize', 20);
xlabel('x', 'FontSize', 20);
ylabel('y', 'FontSize', 20);
for i=1:ntarget
%    labelpoints(est.X{i}(1,est.foundIndex{i}) +1,est.X{i}(2,est.foundIndex{i})+1, num2str(i)); 
   text(est.X{i}(1,est.foundIndex{i}) +1,est.X{i}(2,est.foundIndex{i})+1, num2str(i));
end
axis([0,R_max,0,R_max]);
%{
iptsetpref('ImshowBorder','tight');
set(hFig,'Color','white');
print(hFig,'-depsc2','-painters','SIM_Track_Postion.eps');
 !ps2pdf -dEPSCrop SIM_Track_Postion.eps
%}

% save(['MultiTarget_Tracking',datestr(now, 'yyyymmddHHMMss'),'L5DB_1Cycles.mat'], 'truth', 'meas', 'est','uav_travel_distance_k','uav','MC_Results','mdp_cycles','c' );

return;