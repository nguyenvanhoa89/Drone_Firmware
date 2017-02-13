function Pr = friis_with_ref(A,d0, d, Antenna_Gain)
% Friis transmission equation - Calculate power received by one antenna under idealized 
% conditions given another antenna some distance away transmitting a known amount of power [1]
% Syntax:
%    Pr(d) = Pr(d0) - 10*2*log10(d/d0)
%
% In:
%   A: Ref power at d0 distance (dB)
%   d0: ref distance (m)
%   d: distance from the transmitter.  (m)
%
% Out:
%   Pr(d): Receiver power (dB)

% Copyright (c) 2016 Hoa Van Nguyen
%
% This software is distributed under the GNU General Public 
%%
   Pr = A - 10*2*log10(d) + 10*2*log10(d0) +Antenna_Gain' ; % dB
    
end