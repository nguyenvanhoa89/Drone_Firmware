function [pulse_data, current_index] =  Read_Pulses_With_Index(prev_index)
    url = 'http://localhost:8000/pulses/';
    current_index = size(webread([url, num2str(0)]),1);
    pulse_data = webread([url, num2str(prev_index)]);    
%     test = jsonencode(pulse_data{10});
end
