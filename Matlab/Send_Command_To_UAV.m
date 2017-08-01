function Send_Command_To_UAV (uav_pos)
URL = 'http://localhost:8000/';
options = weboptions('MediaType','application/json');%,'RequestMethod','auto','ArrayFormat','json','ContentType','json');
data = webread(URL);
height_base = data.position(3) - data.position(4); % current home height
write_data.x = uav_pos(1); %data.position(1)+0; % in m
write_data.y = uav_pos(2); %data.position(2) + 0; % in m
write_data.alt = height_base + uav_pos(3);%data.position(3)+3; % in m
write_data.yaw = uav_pos(4); % in rad uav(4,i), if yaw = 7 --> No update yaw angle
webwrite(URL,write_data,options);
end