URL = 'http://localhost:8000/';
options = weboptions('MediaType','application/json');%,'RequestMethod','auto','ArrayFormat','json','ContentType','json');
data = webread(URL);
write_data.x = data.position(1) + 0.001;
write_data.y = data.position(2) + 0.001;
write_data.alt = data.position(3);
response = webwrite('http://localhost:8000',write_data,options);