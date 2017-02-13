function Action_Sets= UAV_Control_Sets_With_Cycles(p0,N_theta, dt, vu, theta_max, mdp_cycle)
% UAV Action Sets
%   UAV_Action_Sets = (X_u, Y_u, Phi_u)
%       X_u: UAV in X-axis
%       Y_u: UAV in Y-axis
%       Phi_u: UAV head-tail direction
% [In]:
%   p0 : Initial UAV Action_Setsition
%   M: number of measurement
%   dt: Sampling cycle (s)
%   vu: UAV vecolocity (m/s)
% [Out]:
%   Action_Sets: UAV Action_Setsition in 2D dimension
%
% Rev0: Nov 4th 2016
% Copyright (c) 2016 Hoa Van Nguyen
%
% This software is distributed under the GNU General Public 
%%
% theta = p0(4);
% clear, clc, close all;
% p0 = [0;0;20;0];
% theta_max = pi/6; % max rotate angle (must less than pi) % current best 5*pi/6; pi/2 not work well
% N_theta = 1; %2 is current best
% vu = 5; % m/s
% dt = 1;
% mdp_cycle = 5;

theta0 = theta_max / N_theta;
cycle_max = round(pi/theta_max); % lower bound
l_max = cycle_max * N_theta;
index = 1;
Action_Sets =  repmat(p0,1,mdp_cycle * (2*l_max+1)+1);
for l = -l_max:1 : l_max
    theta = mod(p0(4) + l * theta0,2*pi);
    needed_cycle = ceil(abs(l * theta0/theta_max));
    if needed_cycle >= mdp_cycle
        index = index + 1;
        Action_Sets(:,index) = [p0(1:3); theta];
    else
%         for j = 1 : (mdp_cycle - needed_cycle)
%             index = index + 1;
%             Action_Sets(:,index) = [p0(1) + j * dt * vu * sin(theta); p0(2) + j * dt * vu * cos(theta); p0(3); theta];
%         end
        j = mdp_cycle - needed_cycle;
        index = index + 1;
        Action_Sets(:,index) = [p0(1) + j * dt * vu * sin(theta); p0(2) + j * dt * vu * cos(theta); p0(3); theta];
    end
end
Action_Sets = unique(Action_Sets','rows')';
end