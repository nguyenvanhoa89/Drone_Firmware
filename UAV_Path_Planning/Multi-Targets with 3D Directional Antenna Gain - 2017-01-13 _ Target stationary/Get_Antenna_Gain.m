function [gain,ThetaInDegrees] = Get_Antenna_Gain(target_pos, uav_pos,gain_angle)
% target_pos = xkp1;
% size(target_pos,2);
if size(target_pos,2) > size(uav_pos,2)
   uav_pos = repmat(uav_pos,1,size(target_pos,2));
else
   target_pos = repmat(target_pos,1,size(uav_pos,2));
end

% gain_angle = load('Gain_Angle_Table.txt');
% gain_angle = load('3D_Directional_Gain_2Yagi_Element.txt'); % Theta	Phi	VdB	HdB	TdB
% gain_max = max(gain_angle(:,2));
v1 = target_pos(1:2,:) - uav_pos(1:2,:);
v2 = [cos(uav_pos(4,:)); sin(uav_pos(4,:))];
% v2 = theta;


x1 = v1(1,:); y1 = v1(2,:);
x2 = v2(1,:); y2 = v2(2,:);
PhiInDegrees=mod(atan2d(x1.*y2-y1.*x2,dot(v1,v2)) + 360,360) ;
resolution = 15;
PhiInDegrees = floor(PhiInDegrees/resolution) * resolution;
ThetaInDegrees = acosd(abs(target_pos(3,:) -uav_pos(3,:))./sqrt(sum((target_pos - uav_pos(1:3,:)).^2)));
ThetaInDegrees = floor(ThetaInDegrees/resolution) * resolution;
% if ThetaInDegrees < 0 
%     ThetaInDegrees = 360 + ThetaInDegrees;
% end
% ThetaInDegrees = atan2d(sqrt(sum(cross(v1,v2).^2)),dot(v1,v2));
% ThetaInDegrees=acos(v1.*v2)*180/pi;
gain = gain_angle((360/resolution+1) *(ThetaInDegrees/resolution)+PhiInDegrees/resolution+1,5);%/gain_max;
% foundIndex = gain_angle(:,1) == ThetaInDegrees & gain_angle(:,2) == PhiInDegrees;
% Total_gain =  repmat(gain_angle(:,5),1,size(ThetaInDegrees,2));
% gain = Total_gain(foundIndex);
% target_pos = [150;100;0];
% uav_pos = [30;0;50];
% ThetaInDegrees = atan2d(sqrt(sum(cross(target_pos,uav_pos).^2)),dot(target_pos,uav_pos));
% gain = gain_angle(round(ThetaInDegrees'/15)+1,2)/gain_max;
end