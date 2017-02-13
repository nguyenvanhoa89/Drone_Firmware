URL = 'http://localhost:8000/';
options = weboptions('MediaType','application/json');%,'RequestMethod','auto','ArrayFormat','json','ContentType','json');
write_count = zeros(1,k);
real_uav = zeros(4,k*2);
real_uav_cycle = 1;
for i=2:k-7
    if mod(i,7) == 0 || i == 2
        tic;
        if i ==2 
            j = i + 5;
        else
            j = i + 7;
        end
    %   write_count(i) = 0;
        data = webread(URL);
        prev_uav = [data.position(1:2);data.position(4); data.heading /180*pi]; 
        Send_Command_To_UAV (uav(:,j));
        while norm(prev_uav(1:3) - uav(1:3,j)) > 1 && write_count(i) < 15
            real_uav_cycle = real_uav_cycle + 1;
            write_count(i) = write_count(i)+ 1;
            data = webread(URL);
            prev_uav = [data.position(1:2);data.position(4); data.heading /180*pi]; 
            real_uav(:,real_uav_cycle) = prev_uav;
            pause(1);
        end
        Command_execute_time(i) = toc;
    end
end
%% Send to last k position then go home
prev_uav = [data.position(1:2);data.position(4); data.heading /180*pi];
Send_Command_To_UAV (uav(:,k));
while norm(prev_uav(1:3) - uav(1:3,k)) > 1 && write_count(k) < 15
    real_uav_cycle = real_uav_cycle + 1;
    write_count(k) = write_count(k)+ 1;
    data = webread(URL);
    prev_uav = [data.position(1:2);data.position(4); data.heading /180*pi]; 
    real_uav(:,real_uav_cycle) = prev_uav;
    pause(1);
end
Send_Command_To_UAV ([50;100;30;0]);

write_count = write_count(write_count>0);
Command_execute_time = Command_execute_time(Command_execute_time>0);
mean(write_count)
function read_pulses_with_index()
    url = 'http://localhost:8000/pulses/';
    initial_length = size(webread([url, num2str(0)]),1);
    test_count  = 1;
    while test_count < 20
        current_index = size(webread([url, num2str(0)]),1) - initial_length;
        pause(1);
        pulse_data{test_count} = webread([url, num2str(initial_length +current_index)]);
        test_count = test_count + 1;
    end
%     test = jsonencode(pulse_data{10});
end

function read_pulses_with_index1()
pulse_index = zeros(1,60);
clear pulse_data;
pulse_index(1) = size(webread([url, num2str(0)]),1);

for i = 2:60
   pause(1);
   [pulse_data{i}, pulse_index(i)] =  Read_Pulses_With_Index(pulse_index(i-1)) ;
end
end

function read_latest_pulse()
    url = 'http://localhost:8000/latestpulses';
    
%     test_count  = 1;
%     while test_count < 20
%         data = webread(url);
%         current_timestamp = data.pulse.timestamp.seconds;
%         current_index = size(webread([url, num2str(0)]),1) - initial_length;
%         pause(0.1);
%         pulse_data{test_count} = webread(url);
%         test_count = test_count + 1;
%     end
%     test = jsonencode(pulse_data{10});
end
%% Rotate UAV
%{
URL = 'http://localhost:8000/';
options = weboptions('MediaType','application/json');
data = webread(URL);
prev_uav = [data.position(1:2);data.position(4); data.heading /180*pi]; 
Send_Command_To_UAV ([prev_uav(1:3); 0]);
pause(5);
i=1;
real_uav_cycle= 0;
write_count = zeros(1,100);
while i <= 10
    data = webread(URL);
    prev_uav = [data.position(1:2);data.position(4); data.heading /180*pi]; 
    current_heading = mod(i*5*pi/6,2*pi);
    Send_Command_To_UAV ([prev_uav(1:3); current_heading]);
    while norm(prev_uav(4) -current_heading) > 0.1 && write_count(i) < 15
        real_uav_cycle = real_uav_cycle + 1;
        write_count(i) = write_count(i)+ 1;
        data = webread(URL);
        prev_uav = [data.position(1:2);data.position(4); data.heading /180*pi]; 
        pause(1);
    end
    i = i + 1;
end

%}


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
