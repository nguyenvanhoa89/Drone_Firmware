function [Action_Sets, Cost_Sets]= UAV_Control_Sets(p0,Np,N_theta, dt, vu, theta_max)
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
theta0 = pi/N_theta;
index = 1;
l_max = round(theta_max* N_theta/pi);
Action_Sets = zeros(3, Np * (2*l_max+1)+1);
Cost_Sets = zeros(1, Np * (2*l_max+1)+1);
Action_Sets(:,index) = p0;
Cost_Sets(index) = 0;
for j = 1 : Np
    for l = -l_max:1 : l_max
        index = index + 1;
        Cost_Sets(index) = -j;
        Action_Sets(:,index) = [p0(1) + j * dt * vu * cos(l * theta0); p0(2) + j * dt * vu * sin(l * theta0); p0(3)];
    end
end

end