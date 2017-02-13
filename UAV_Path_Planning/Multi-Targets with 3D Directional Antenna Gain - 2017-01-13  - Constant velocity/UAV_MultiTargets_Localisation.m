%% clear memory, screen, and close all figures
tic;
clear, clc, close all;

%% Process equation x[k] = sys(k, x[k-1], u[k]);
nx = 5;  % number of states [x y vx vy z]
nuav = 4;
dt = 1; % second
q = 1; % Current work for small q = 1 only. Higher not good
nu = 2;   % dimension of process noise
ntarget = 3;
uav0 = [0;0;20;0];

R_max = 500;
x0 = [R_max* rand; R_max * rand; 0; -q + 2*q*rand;-q + 2*q*rand];  
%% Initial variable
T = 200;
Time = 1;
pf.Ns = 3000;             % number of particles
Ms = 100; % 100 is current best
alpha = 0.5;
Area = [0 0 uav0(3);R_max R_max uav0(3)]';
theta_max = 5*pi/6; % max rotate angle (must less than pi) % current best 5*pi/6
N_theta = 12; %12 is current best
Np = 1; % max velocity = Np * vu (m/s) % 2 with 5m/s is current best
vu = 10; % m/s
RSS_Threshold = -125; % dB
plot_box = 1;
%% dynamical model parameters (CV model) [x y  z vx vy]
F = [diag([1 1]) zeros(2,1) diag([dt dt]) ;
     diag([0 0]) diag([1 1])  zeros(2,1);
     zeros(1,4)  1                       ];
% Q = q^2 * [dt^4/4*diag([1 1]) zeros(2,1) dt^3/2 * diag([1 1]);
%             zeros(1,5);            
%             dt^3/2*diag([1 1]) zeros(2,1)  dt^2* diag([1 1]) ;];
sys_noise.B0 = [dt^2/2; dt];
sys_noise.B =  [sys_noise.B0 zeros(2,1); zeros(1,2); zeros(2,1) sys_noise.B0; ];
% V = q*sys_noise.B*randn(size(sys_noise.B,2),200);
sys = @(k, F, x, uk) F*x + uk; % random walk object
%% PDF of process noise and noise generator function
% sigma_u = q^nx * eye(nx);
sigma_u = q;%^2 * eye(nu);
% Q = [];
% sigma_u = Q;
% p_sys_noise   = @(u) mvnpdf(u, zeros(1,nu), sigma_u);
gen_sys_noise = @(u) mvnrnd(zeros(1,nu),sigma_u,1)';         % sample from p_sys_noise (returns column vector)
gen_sys_noise = @(u) q*sys_noise.B*randn(size(sys_noise.B,2),1);
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




%% Generate true target state-space model
truth.X = cell(ntarget,1);
x = zeros(nx,T);
for i=1:ntarget
    x(:,1) = [R_max * rand/1.2 ; R_max * rand/1.2 ; 0; -q + 2*q*rand;-q + 2*q*rand];
    truth.X{i} = x(:,1);
   for k=2:T
%         x(:,k) = sys(k,F, x(:,k-1), gen_sys_noise()); 
        x(:,k) = sys(k,F, x(:,k-1), 0); 
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
gain_angle = load('3D_Directional_Gain_2Yagi_Element.txt'); % Theta	Phi	VdB	HdB	TdB
% obs = @(k, x, vk,uav) d(x,uav)     + vk ;     
obs = @(k, x, vk,uav,gain_angle) friis(Pt, Gt, Gr, lambda, L, d(x(1:3,:),uav(1:3,:)),Get_Antenna_Gain(x, uav,gain_angle))     + vk ;     % (returns column vector)
% [obsmin, obsmax] = Calculate_Mix_Max_Friss(Pt,Gt, Gr, L, R_max, uav0(3), err_loc, phi);

% obs = @(k, xk, vk) xk(1)^2/20 + vk;                  % (returns column vector)
% obs = @(k, x, vk) sin(x(1)) + vk ;                  % (returns column vector)


%% UAV Initialization
uav = zeros(nuav,T);
% uav(:,1) = uav0;
% uavtraj = uav_3d_circle([uav0; 0 ], T,dt, vu);
% uav = uavtraj(1:3,:);
uav(:,2) = uav0;
fprintf('Filtering with PF...');
% fprintf('Press any key to see the result');
% pause 
%% Separate memory

pf.k               = 1;                   % initial iteration number
% pf.Ns              = 5000;                 % number of particles
pf.w               = ones(pf.Ns, T)/pf.Ns;     % weights
pf.particles       = [R_max * rand(pf.Ns,1) R_max * rand(pf.Ns,1) zeros(pf.Ns,1) sigma_u*rand(pf.Ns,1) sigma_u*rand(pf.Ns,1)]'; % particles
% pf.gen_x0          = gen_x0;              % function for sampling from initial pdf p_x0
pf.gen_x0          = [R_max * rand(pf.Ns,1) R_max * rand(pf.Ns,1) zeros(pf.Ns,1) sigma_u*rand(pf.Ns,1)  sigma_u*rand(pf.Ns,1)]';
pf.p_yk_given_xk   = p_yk_given_xk;       % function of the observation likelihood PDF p(y[k] | x[k])
pf.gen_sys_noise   = gen_sys_noise;       % function for generating system noise
pf.gen_obs_noise   = gen_obs_noise;
pf.sigma_v         = sigma_v;
pf.sigma_u         = sigma_u;
pf.RSS_Threshold   = RSS_Threshold;
pf.R_max           = R_max;
pf.gain_angle      = gain_angle;
pf.F               = F;
pf.sys_noise       = sys_noise;
%pf.p_x0 = p_x0;                          % initial prior PDF p(x[0])
%pf.p_xk_given_ xkm1 = p_xk_given_xkm1;   % transition prior PDF p(x[k] | x[k-1])
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
measurement = zeros(1,ntarget);
Reward = zeros(1,ntarget);
for k = 2:T
    [UAV_Sets,Cost_Sets] = UAV_Control_Sets(uav(:,k),Np,N_theta,dt,vu,theta_max);
    for i=1:ntarget 
       measurement(i) = obs(k, truth.X{i}(1:3,k),   gen_obs_noise(),uav(:,k),pf.gain_angle);
       if isempty(est.foundTargetList(i == est.foundTargetList))
           est.pf{i}.k = k;
           if measurement(i) > RSS_Threshold
               meas.ValidZCount{i} = meas.ValidZCount{i} + 1;
               meas.Z{i}(k)  = measurement(i);
               [est.X{i}(:,k), est.pf{i}] = bootstrap_filter (k, est.pf{i}, sys, obs, meas.Z{i}(k), uav(:,k));
               % Calculate next UAV position
               [meas.UAV{i}(:,k),meas.Reward{i}(k)]  = Control_Vector_Selection (k, est.pf{i},sys,obs, UAV_Sets, Ms, alpha, Area);
%                if meas.ValidZCount{i}>5 && abs(std(meas.Z{i}(k-5:k))) < 2*sigma_v && abs(mean(meas.Z{i}(k-5:k)) - RSS_Threshold) < 3* sigma_v &&   norm(std(est.pf{i}.particles(:,:,k),0,2)) < sigma_v %&& abs( obs(k, est.X{i}(:,k),   gen_obs_noise(),uav(:,k)) - measurement(i)) < sigma_v
               if det(cov(est.pf{i}.particles(1:2,:,k)')) < pf.Ns
                   est.foundX{i} = est.X{i}(:,k);
                   est.foundIndex{i} = k;
                   est.foundTargetList = [est.foundTargetList i];
               end
           else
               est.X{i}(:,k) = sys(k, F, est.X{i}(:,k-1), gen_sys_noise());
               est.pf{i}.w(:,k) =  est.pf{i}.w(:,k-1);
               est.pf{i}.particles(:,:,k) = est.pf{i}.particles(:,:,k-1);
               [meas.UAV{i}(:,k),meas.Reward{i}(k)]  = Control_Vector_Selection (k, est.pf{i},sys,obs, UAV_Sets, Ms, alpha, Area);
    %            meas.Reward{i} = -1;
               measurement(i) = RSS_Threshold;
           end
           
       else
           meas.Reward{i}(k) = -1; % Min value to skip
           measurement(i) = -1e3; % Min value to skip
       end
       Reward(i) = meas.Reward{i}(k); %% Dummy variable to store current reward function
       if Reward(i) == -1e30 % If all of way points out of pre-defined area, back to previous position
           meas.UAV{i}(:,k) = uav(:,k-1);
       end
    end
%     [~,best_u] = max(Reward,[],2);
    [~,best_u] = max(measurement,[],2);
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
uav_travel_distance_k
time_PF = toc;
fprintf('PF Execution Time         :%5.2fs\n',time_PF);
%% Plot
ntarget = size(truth.X,1);
R_max = est.pf{1}.R_max;
k = max([est.foundIndex{:}]);
c = get(gca,'ColorOrder');
% c = rand(ntarget+1,3);
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
title('Position estimation with Particle filter.');
xlabel('x');
ylabel('y');
for i=1:ntarget
%    labelpoints(est.X{i}(1,est.foundIndex{i}) +1,est.X{i}(2,est.foundIndex{i})+1, num2str(i)); 
   text(est.X{i}(1,est.foundIndex{i}) +2,est.X{i}(2,est.foundIndex{i})+2, num2str(i));
end
axis([0,R_max,0,R_max]);
% axis([0,300,0,300]);

save(['MultiTarget_Estimate',datestr(now, 'yyyymmddHHMMss'),'.mat'], 'truth', 'meas', 'est','uav_travel_distance_k','uav' );

return;