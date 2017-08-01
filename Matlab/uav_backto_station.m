function pos = uav_backto_station(pstart, pend,theta_max, dt,vu)
% UAV 2D Circle Sampling Position with constant altitude assumption
%   UAV_POS = (X_u, Y_u, Phi_u)
%       X_u: UAV in X-axis
%       Y_u: UAV in Y-axis
%       Phi_u: UAV head-tail direction
% [In]:
%   p0 : Initial UAV Position
%   M: number of measurement
%   dt: Sampling cycle (s)
%   vu: UAV vecolocity (m/s)
% [Out]:
%   pos: UAV Position in 2D dimension
%
% Rev0: Nov 4th 2016
% Copyright (c) 2016 Hoa Van Nguyen
%
% This software is distributed under the GNU General Public 
%%
% pstart = [200;300;20;1.8*pi];
% pend = [0;0;20;0];
v1 = pend(1:2,:) - pstart(1:2,:) ;
v2 = [cos(pstart(4,:)); sin(pstart(4,:))];
% v2 = theta;
x1 = v1(1,:); y1 = v1(2,:);
x2 = v2(1,:); y2 = v2(2,:);
theta=atan2(x1.*y2-y1.*x2,dot(v1,v2));
Cycle = ceil(norm(pstart(1:2) - pend(1:2)) / (dt * vu)) + ceil(abs(theta/theta_max));
pos = zeros(4,Cycle+1);
% theta_target = pi - atan2(pstart(2),pstart(1));
for i=1:ceil(abs(theta/Cycle))
    if i == 1
        pos(:,i) = [pstart(1:3); pstart(4) - theta_max]; 
    else
        pos(:,i) = [pos(1:3,i-1); pos(4,i-1) - theta_max]; 
    end
    theta = theta - theta_max;
end

% pos(:,1) = pstart;
% i = 1;
% while theta > theta_max
%     i = i +1; 
%     pos(:,i) = [pos(1:3,i-1); pos(4,i-1) - theta_max]; 
%     theta = theta - theta_max;
% end
pos(:,i+1) = [pos(1:3,i); pos(4,i) - theta];
% T = i+1+ floor(abs((pos(1,i+1) - pend(1)) / (dt*vu*cos(pos(4,i+1)))));

% alpha = 2*pi/(M*dt); % rad/s - UAV turn rate
uav = @(x) [x(1) + dt*vu*cos(x(4)); 
            x(2) + dt*vu*sin(x(4)); 
            x(3)
            x(4) ];

% pos = zeros(4,M);
% pos(:,1) = p0;
for j=i+2:Cycle
    pos(:,j) = uav(pos(:,j-1));
    if pos(1,j) < 0 || pos(2,j) < 0
        pos(:,j) = pos(:,j-1);
        break;
    end
end
pos(:,Cycle+1) = pend;
end