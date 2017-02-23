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
% ntarget = 2;
% target_frequency = [150,148,152];
% target_frequency = [1,2];
target_frequency = [150130000,150411000,150201000];
ntarget = size(target_frequency,2);
Obs_From_Tele = 1; % Set this para to 0 if want to run locally
uav0 = [0;0;20;0];
sys = @(k, x, uk) x + uk; % random walk object
R_max = 100;
x0 = [R_max * (2*rand -1); R_max * (2*rand -1); 0];  
%% Initial variable
T = 500; % 15 minutes is max
% Time = 1;
pf.Ns = 3000;             % number of particles
Ms = 100; % 100 is current best
alpha = 0.5;
Area = [-R_max -R_max uav0(3);R_max R_max uav0(3)]';
theta_max = pi/6; % max rotate angle (must less than pi) % current best 5*pi/6; pi/2 not work well
N_theta = 1; %12 is current best
% Np = 1; % max velocity = Np * vu (m/s) % 2 with 5m/s is current best
vu = 5; % m/s
RSS_Threshold = 20*log10(1/256) - 102; % dB Ref: -25, Normal: -125
plot_box = 1;
%% PDF of process noise and noise generator function
% sigma_u = q^nx * eye(nx);
sigma_u = q^2 * [1 1 0];
Q = [];
nu = size(sigma_u,2); % size of the vector of process noise
p_sys_noise   = @(u) mvnpdf(u, zeros(1,nu), sigma_u);
gen_sys_noise = @(u) mvnrnd(zeros(1,nu),sigma_u,1)';         % sample from p_sys_noise (returns column vector)
%% PDF of observation noise and noise generator function
sigma_v = 7^2;
nv =  size(sigma_v,1);  % size of the vector of observation noise
p_obs_noise   = @(v) mvnpdf(v, zeros(1,nv), sigma_v);
gen_obs_noise = @(v) mvnrnd(zeros(1,nv),sigma_v,1)';         % sample from p_obs_noise (returns column vector)

%% Initial PDF
% p_x0 = @(x) normpdf(x, 0,sqrt(10));             % initial pdf
gen_x0 = @(x) [R_max* (2*rand -1) R_max* (2*rand -1) 0]';               % sample from p_x0 (returns column vector)

%% Observation likelihood PDF p(y[k] | x[k])
% (under the suposition of additive process noise)
p_yk_given_xk = @(k, yk, xk, uavk) p_obs_noise(yk - obs(k, xk, 0, uavk));

%% Observation equation y[k] = obs(k, x[k], v[k]);
ht = 1;
hr = 20;
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
% obs = @(k, x, vk,uav,gain_angle) friis_2model(Pt, Gt, Gr, lambda, L, x,uav,Get_Antenna_Gain(x, uav,gain_angle))     + vk ;     % (returns column vector)
% For ref model
A_ref = -7 - 0.454*32 ; d0 = 1; % (m) 12.67, change to other to test
% A_ref = -25.6603 ; d0 = 1; % (m) 12.67, change to other to test
obs = @(k, x, vk,uav,gain_angle)  friis_with_ref(A_ref,d0, d(x,uav(1:3,:)),Get_Antenna_Gain(x, uav,gain_angle))     + vk ;     % (returns column vector)
obs_real = @(amplitude, gain) 20 *log10(amplitude) - 0.454 * gain ;
%% UAV Initialization
uav = zeros(nuav,T);
URL = 'http://localhost:8000/';
options = weboptions('MediaType','application/json');
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
pf.mean_rss_std    = zeros(1,T);
%pf.p_x0 = p_x0;                          % initial prior PDF p(x[0])
%pf.p_xk_given_ xkm1 = p_xk_given_xkm1;   % transition prior PDF p(x[k] | x[k-1])


%% Main Program

Time = 1;
% mdp_cycles = [1,3,5,7,10];
% mdp_cycles = [5,7,10,13,15];
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
        uav = zeros(nuav,T);
%         MC_Results.uav_travel_distance(time,mdp_cycle) = 0;
        est.foundTargetList = [];
        uav_travel_distance_k = 0;
        measurement = RSS_Threshold* ones(1,ntarget);
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
        meas.UAV = zeros(nuav,T);
        % Initial measurement for each target estimation
        for i=1:ntarget 
            meas.Z{i} = zeros(1,T);
            meas.ValidZCount{i} = 0;
%             meas.UAV{i} = zeros(nuav,T);
            meas.Reward{i} = zeros(1,T);
        end
        %% Intialize pulse
        clear Pulse;
        url = 'http://localhost:8000/pulses/';
        Pulse.pulse_index = zeros(1,T);
        Pulse.pulse_index(1) = size(webread([url, num2str(0)]),1);
        pause(1);
        Pulse.pulse_data = cell(T,1);
        Pulse.pulse_struct = cell(T,1);
        Pulse.pulse_freq = cell(T,1);
        Pulse.pulse_signal_strength = cell(T,1);
        Pulse.pulse_rss = cell(T,1);
        Pulse.gain = cell(T,1);
        % Main program for estimation
        for k = 2:T
            fprintf('Iteration = %d/%d\n',k,T);
            data = webread(URL);
            if Obs_From_Tele == 1  
                [Pulse.pulse_data{k}, Pulse.pulse_index(k)] =  Read_Pulses_With_Index(Pulse.pulse_index(k-1)) ;
                 Pulse.pulse_struct{k} = [Pulse.pulse_data{k}.pulse];
                 Pulse.pulse_freq{k} = fliplr([Pulse.pulse_struct{k}(:).freq]); % flip to get latest data first
                 Pulse.pulse_signal_strength{k} = fliplr([Pulse.pulse_struct{k}(:).signal_strength]);
                 Pulse.pulse_gain{k} = fliplr([Pulse.pulse_struct{k}(:).gain]);
                 Pulse.pulse_rss{k} = obs_real(Pulse.pulse_signal_strength{k},Pulse.pulse_gain{k});
                 pulse_rss = Pulse.pulse_rss{k} ;
                % update measurement from pulse data
                for i=1:ntarget
                   if ~isempty(pulse_rss(target_frequency(i) == Pulse.pulse_freq{k}))
                       [~,indx] = max(target_frequency(i) == Pulse.pulse_freq{k},[],2); % get latest info
                        measurement(i) = pulse_rss(indx);
                        if Pulse.pulse_signal_strength{k}(indx) < 1/256 || Pulse.pulse_signal_strength{k}(indx) > 0.95 || ~ isempty(est.foundTargetList(i == est.foundTargetList))
                           measurement(i) = RSS_Threshold;
                        end
                   end
                end
            else
                % Get update measurement from simulated data
                for i=1:ntarget 
                   measurement(i) = obs(k, truth.X{i}(:,k),   gen_obs_noise(),uav(:,k),pf.gain_angle);
                   if measurement(i) < RSS_Threshold || ~ isempty(est.foundTargetList(i == est.foundTargetList))
                       measurement(i) = RSS_Threshold;
                   end
                end
            end
            uav(:,k) = [data.position(1:2);data.position(4); data.heading /180*pi]; 
            
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
            if mod(k,mdp_cycles(mdp_cycle))==0 || k == 2 % Update way point every 5 seconds only
                UAV_Sets = UAV_Control_Sets_With_Cycles(uav(:,k),N_theta, dt, vu, theta_max, mdp_cycles(mdp_cycle));
                [meas.UAV(:,k),~]  = Control_Vector_Selection (k, est.pf{best_u},sys,obs, UAV_Sets, Ms, alpha, Area);
                Send_Command_To_UAV (meas.UAV(:,k));
            end
        %     [~,best_u] = max(Reward,[],2);
%             uav(:,k+1) = meas.UAV{best_u}(:,k);
            uav_travel_distance_k = uav_travel_distance_k + norm(uav(:,k) - uav(:,k-1));
            if size(est.foundTargetList,2) == ntarget || (norm(uav(:,k))/(dt*vu) + k) > T
               break; 
            end
            pause(1);
        end
        for i=1:ntarget
            if isempty(est.foundTargetList(i == est.foundTargetList))
               est.foundX{i} = est.X{i}(:,k);
               est.foundIndex{i} = k;
            end
            est.RMSFound{i} = d( est.foundX{i}, truth.X{i}(:,est.foundIndex{i})) ;
        end
        Send_Command_To_UAV (uav0); % back to station
        data = webread(URL);
        prev_uav = [data.position(1:2);data.position(4); data.heading /180*pi]; 
        while norm(prev_uav(1:3) - uav0(1:3)) > 0.5
            k = k+1;
            data = webread(URL);
            prev_uav = [data.position(1:2);data.position(4); data.heading /180*pi]; 
            uav(:,k) = prev_uav;
            pause(1);
        end
        if Obs_From_Tele == 1
            truth.X{1} = [400;50;0];
            truth.X{2} = [50;100;0];
            truth.X{3} = [200;300;0];
            for i=1:ntarget
                x(:,1) = truth.X{i};                
               for k=2:T
                    x(:,k) = sys(k, x(:,k-1), [0;0;0]); 
                    truth.X{i}= [truth.X{i} x(:,k)];
               end   
            end
        end
        Plot_Target_Estimated_Position (truth, est, uav);
        Send_Command_To_UAV (uav0);
        pause(10);
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
% v1 = pend(1:2,:) - pstart(1:2,:) ;
% v2 = [cos(pstart(4,:)); sin(pstart(4,:))];
% % v2 = theta;
% x1 = v1(1,:); y1 = v1(2,:);
% x2 = v2(1,:); y2 = v2(2,:);
% theta=atan2(x1.*y2-y1.*x2,dot(v1,v2));
% Cycle = ceil(norm(pstart(1:2) - pend(1:2)) / (dt * vu)) + ceil(abs(theta/theta_max));
% uav(:,k+1:k+Cycle+1) = uav_backto_station(pstart, pend,theta_max, dt,vu);
est.foundIndex
est.RMSFound
uav_travel_distance_k = uav_travel_distance_k + norm(pstart(1:2) - pend(1:2))
time_PF = toc;
fprintf('PF Execution Time         :%5.2fs\n',time_PF);


%{
iptsetpref('ImshowBorder','tight');
set(hFig,'Color','white');
print(hFig,'-depsc2','-painters','SIM_Estimate_Postion.eps');
 !ps2pdf -dEPSCrop SIM_Estimate_Postion.eps
%}

% save(['MultiTarget_Estimate',datestr(now, 'yyyymmddHHMMss'),'L5DB_5Cycles.mat'], 'truth', 'meas', 'est','uav_travel_distance_k','uav','MC_Results','mdp_cycles' );

return;