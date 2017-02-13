%% clear memory, screen, and close all figures
tic;
clear, clc, close all;

%% Process equation x[k] = sys(k, x[k-1], u[k]);
nx = 3;  % number of states
nuav = 4;
dt = 1; % second
q = 1;
ntarget = 3;
uav0 = [0;0;20;0];
sys = @(k, x, uk) x + uk; % random walk object
R_max = 500;
x0 = [R_max * rand; R_max * rand; 0];  
%% Initial variable
T = 200;
Time = 1;
pf.Ns = 3000;                 % number of particles
Ms = 100; % 100 is current best
alpha = 0.5;
Area = [0 0 uav0(3);R_max R_max uav0(3)]';
theta_max = 5*pi/6; % max rotate angle (must less than pi)
N_theta = 12; %12 is current best
Np = 1; % max velocity = Np * vu (m/s) % 2 with 5m/s is current best
vu = 10; % m/s
RSS_Threshold = -130; % dB
plot_box = 1;

%% Generate true target state-space model
truth.X = cell(ntarget,1);
for i=1:ntarget
    x(:,1) = [R_max * rand; R_max * rand; 0]; 
    truth.X{i} = x(:,1);
   for k=2:T
        x(:,k) = sys(k, x(:,k-1), 0); 
        truth.X{i}= [truth.X{i} x(:,k)];
   end   
end

%% Observation equation y[k] = obs(k, x[k], v[k]);
ht = 1;
hr = 50;
PtW = 0.5e-3;
Pt = 10*log10(PtW); %dBm
f = 173e6;
c = physconst('lightspeed');
lambda = c/f;
Gt = 0;   %dBm
Gr = -15; %dBm
L = 12; %dBm 
d = @(x,uav) sqrt(sum((x-uav).^2)); % distance between UAV and target
ny = 1;                                           % number of observations
err_loc = 0.2;
phi = [2 3];
% obs = @(k, x, vk,uav) d(x,uav)     + vk ;     
obs = @(k, x, vk,uav) friis(Pt, Gt, Gr, lambda, L, d(x,uav(1:3,:)),Get_Antenna_Gain(x, uav))     + vk ;     % (returns column vector)
% [obsmin, obsmax] = Calculate_Mix_Max_Friss(Pt,Gt, Gr, L, R_max, uav0(3), err_loc, phi);

% obs = @(k, xk, vk) xk(1)^2/20 + vk;                  % (returns column vector)
% obs = @(k, x, vk) sin(x(1)) + vk ;                  % (returns column vector)

%% PDF of process noise and noise generator function
% sigma_u = q^nx * eye(nx);
sigma_u = q^2 * [1 1 0];
Q = [];
nu = size(sigma_u,2); % size of the vector of process noise
p_sys_noise   = @(u) mvnpdf(u, zeros(1,nu), sigma_u);
gen_sys_noise = @(u) mvnrnd(zeros(1,nu),sigma_u,1)';         % sample from p_sys_noise (returns column vector)
%% PDF of observation noise and noise generator function
sigma_v = 10;
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
x = zeros(nx,T);  y = zeros(ny,T);
u = zeros(nu,T);  v = zeros(nv,T);
uav = zeros(nuav,T);

%% Simulate system
% uav(:,1) = uav0;
% uavtraj = uav_3d_circle([uav0; 0 ], T,dt, vu);
% uav = uavtraj(1:3,:);
uav(:,2) = uav0;
                                % initial state
u(:,1) = zeros(nu,1);                               % initial process noise
v(:,1) = gen_obs_noise(sigma_v);          % initial observation noise
x(:,1) = x0;
y(:,1) = obs(1, x0, v(:,1),uav0);
for k = 2:T
   % here we are basically sampling from p_xk_given_xkm1 and from p_yk_given_xk
   u(:,k) = gen_sys_noise();              % simulate process noise
   v(:,k) = gen_obs_noise();              % simulate observation noise
%    if k <= 240
%        uav(:,k) = uav(:,k-1) + [0; vu * dt; 0];
%    elseif k <= 260
%        uav(:,k) = uav(:,k-1) + [vu * dt; 0; 0];
%    else 
%        uav(:,k) = uav(:,k-1) + [0; -vu * dt; 0];
%    end
   
   x(:,k) = sys(k, x(:,k-1), 0);     % simulate state
%    y(:,k) = obs(k, x(:,k),   v(:,k),uav(:,k));     % simulate observation
end
% plot(x(1,:),x(2,:),'.');
% legend('Real trajectory');
% title('Position');
% xlabel('x');
% ylabel('y');
fprintf('Filtering with PF...');
% fprintf('Press any key to see the result');
% pause 
%% Separate memory
xh = zeros(nx, T); %xh(:,1) = [100;100;0];
yh = zeros(ny, T); yh(:,1) = obs(1, xh(:,1), 0, uav0);

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
%pf.p_x0 = p_x0;                          % initial prior PDF p(x[0])
%pf.p_xk_given_ xkm1 = p_xk_given_xkm1;   % transition prior PDF p(x[k] | x[k-1])
RMS_Distance(1) = d(xh(:,1), x(:,1)) ;
%% Estimation Initial
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
est.foundTargetList = [];
%% Main Program
uav_travel_distance_k = 0;
for k = 2:T
    [UAV_Sets,Cost_Sets] = UAV_Control_Sets(uav(:,k),Np,N_theta,dt,vu,theta_max);
    for i=1:ntarget 
       measurement = obs(k, truth.X{i}(:,k),   gen_obs_noise(),uav(:,k));
       if isempty(est.foundTargetList(i == est.foundTargetList))
           est.pf{i}.k = k;
           if measurement > RSS_Threshold
               meas.ValidZCount{i} = meas.ValidZCount{i} + 1;
               meas.Z{i}(k)  = measurement;
               [est.X{i}(:,k), est.pf{i}] = bootstrap_filter (k, est.pf{i}, sys, obs, meas.Z{i}(k), uav(:,k));
               % Calculate next UAV position
               [meas.UAV{i}(:,k),meas.Reward{i}(k)]  = Control_Vector_Selection (k, est.pf{i},sys,obs, UAV_Sets, Ms, alpha, Area);
               if meas.ValidZCount{i}>5 && abs(std(meas.Z{i}(k-5:k))) < 2*sigma_v && abs(mean(meas.Z{i}(k-5:k)) - RSS_Threshold) < 3* sigma_v &&   norm(std(est.pf{i}.particles(:,:,k),0,2)) < sigma_v/3 %&& abs( obs(k, est.X{i}(:,k),   gen_obs_noise(),uav(:,k)) - measurement) < sigma_v
                   est.foundX{i} = est.X{i}(:,k);
                   est.foundIndex{i} = k;
                   est.foundTargetList = [est.foundTargetList i];
               end
           else
               est.X{i}(:,k) = sys(k, est.X{i}(:,k-1), 0);
               est.pf{i}.w(:,k) =  est.pf{i}.w(:,k-1);
               est.pf{i}.particles(:,:,k) = est.pf{i}.particles(:,:,k-1);
               [meas.UAV{i}(:,k),meas.Reward{i}(k)]  = Control_Vector_Selection (k, est.pf{i},sys,obs, UAV_Sets, Ms, alpha, Area);
    %            meas.Reward{i} = -1;
           end
           
       else
           meas.Reward{i}(k) = -1;
       end
       Reward{i} = meas.Reward{i}(k); %% Dummy variable to store current reward function
    end
    [~,best_u] = max([Reward{:}],[],2);
    uav(:,k+1) = meas.UAV{best_u}(:,k);
    uav_travel_distance_k = uav_travel_distance_k + norm(uav(:,k+1) - uav(:,k));
    if size(est.foundTargetList,2) == ntarget
       break; 
    end
    
end
for i=1:ntarget
    if isempty(est.foundTargetList(i == est.foundTargetList))
       est.foundX{i} = est.X{i}(:,k);
       est.foundIndex{i} = k;
    end
end
for i=1:ntarget 
   est.RMS{i} = d( est.foundX{i}, truth.X{i}(:,est.foundIndex{i})) ;
end
est.foundIndex
est.RMS
uav_travel_distance_k
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
c = get(gca,'ColorOrder');
hFig = figure(1);
set(hFig, 'Position', [100 500 1000 1000]);
% subplot(2,2,1);
hold on;
for i=1:ntarget
   htruth{i} = plot(truth.X{i}(1,est.foundIndex{i}),truth.X{i}(2,est.foundIndex{i}), 'Color' , c(i,:) , 'Marker' , '+','markersize',10);
   hest{i} = plot(est.X{i}(1,est.foundIndex{i}),est.X{i}(2,est.foundIndex{i}),'Color' , c(i,:) , 'Marker' , '.','markersize',10);
end
huav = plot(uav(1,1:k), uav(2,1:k),'-', 'Color' , c(ntarget+1,:));
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
title('Position estimation with Particle filter.');
xlabel('x');
ylabel('y');
axis([0,R_max,0,R_max]);

% save('MultiTarget_Estimate.mat', 'truth', 'meas', 'est','uav_travel_distance_k' );
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