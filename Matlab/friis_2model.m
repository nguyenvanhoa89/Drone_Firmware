function Pr = friis_2model(Pt, Gt, Gr, lambda, L, x,uav , Antenna_Gain)
% Syntax:
%    Pr = friis_2model(Pt, Gt, Gr, lambda, L, d, ht, hr) (dBm)
%
% In:
%   Pt: Transmitter power (dBm). Pt (dBm) = 10*log10(Pt(mW));
%   Gt: Transmitter gain (dBm)
%   Gr: Receiver gain (dBm)
%   lamda: waveleght of carrier (m). Lamda = c/freq. c = light speed
%   (3x10e8) c = physconst('lightspeed');
%   L: other losses not belong to propagation loss. (dBm)
%   d: distance between the transmitter and receiver  (m)
%   ht: transmitter height (m)
%   hr: receiver height (m)
%
% Out:
%   Pr: Receiver power (dBm)
%
% Friis free space propagation with 2 models:
%
% [1] A. Posch ; S. Sukkarieh, "UAV based search for a radio tagged animal using particle filter". 2. Dec. 2009.
%
% Copyright (c) 2016 Hoa Van Nguyen
%
% This software is distributed under the GNU General Public 
%%
if size(x,2) > size(uav,2)
   uav = repmat(uav,1,size(x,2));
else
   x = repmat(x,1,size(uav,2));
end
d = @(x,uav) sum((x(1:2,:)-uav(1:2,:)).^2); % distance between UAV and target
D = d(x,uav);
ht = x(3,:)+1; 
hr = uav(3,:);
l = sqrt( D+ (ht+hr).^2) -  sqrt(D + (ht-hr).^2);
phi =  2*pi*l/lambda;
Pr = Pt + Gt + Gr - L + Antenna_Gain' - 10*log10(4*pi^2) -10*log10(D) + 20*log10(abs(sin(phi/2))) + 20*log10(lambda) ;
end