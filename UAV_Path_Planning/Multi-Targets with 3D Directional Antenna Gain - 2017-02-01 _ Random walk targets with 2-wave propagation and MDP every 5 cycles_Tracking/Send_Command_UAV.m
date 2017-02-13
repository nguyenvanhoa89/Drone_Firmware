function Send_Command_UAV (uav)
URL = 'http://localhost:8000/';
options = weboptions('MediaType','application/json');%,'RequestMethod','auto','ArrayFormat','json','ContentType','json');
% for i =1:size(uav,2)
write_count = zeros(1,k);
for i=1:5:k
    tic;
%   write_count(i) = 0;
    data = webread(URL);
    base = data.position(3) - data.position(4); % current home height
    prev_uav = [data.position(1:2);data.position(4); data.heading /180*pi];
    write_data.x = uav(1,i); %data.position(1)+0; % in m
    write_data.y = uav(2,i); %data.position(2) + 0; % in m
    write_data.alt = base + uav(3,i);%data.position(3)+3; % in m
    write_data.yaw = uav(4,i); % in rad uav(4,i), if yaw = 7 --> No update yaw angle
    response = webwrite('http://localhost:8000',write_data,options);
    while norm(prev_uav(1:3) - uav(1:3,i)) > 1 && write_count(i) < 15
        write_count(i) = write_count(i)+ 1;
        data = webread(URL);
        prev_uav = [data.position(1:2);data.position(4); data.heading /180*pi];     
        pause(1);
    end
    Command_execute_time(i) = toc;
end
write_count = write_count(write_count>0);
mean(write_count)

%{

latlon1=[47.3977422 8.5455942  ]; 
latlon2=[47.39864584990353 8.546924185890807 ];
%   latlon1=[-43 172];
%   latlon2=[-44  171];
arclen = distance('gc',latlon1,latlon2)*6371;

pos1     = [32.22, 15.09];
pos2     = [32.45, 15.55];
h        = 20;                                 % // altitude                         
SPHEROID = referenceEllipsoid('wgs84', 'm'); % // Reference ellipsoid. You can enter 'km' or 'm'    
[N, E]   = geodetic2ned(latlon1(1), latlon1(2), h, latlon2(1), latlon2(2), h, SPHEROID);
distance = norm([N, E])
 %}
end