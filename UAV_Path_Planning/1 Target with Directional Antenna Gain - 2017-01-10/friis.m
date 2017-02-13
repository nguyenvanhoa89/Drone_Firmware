function Pr = friis(Pt, Gt, Gr, lambda, L, d, Antenna_Gain)
% Friis transmission equation - Calculate power received by one antenna under idealized 
% conditions given another antenna some distance away transmitting a known amount of power [1]
% Syntax:
%    Pr = friis(Pt, Gt, Gr, lambda, L, d)
%
% In:
%   Pt: Transmitter power (dB). Pt (dB) = 10*log10(Pt(W)*1000);
%   Gt: Transmitter gain (dB)
%   Gr: Receiver gain (dB)
%   lamda: waveleght of carrier (m). Lamda = c/freq. c = light speed
%   (3x10e8) c = physconst('lightspeed');
%   L: other losses not belong to propagation loss. L =1 mean no loss (dB)
%   d: distance from the transmitter.  (m)
%
% Out:
%   Pr: Receiver power (dB)
%
% Friis free space propagation model:
%        Pt * Gt * Gr * (lambda^2)
%  Pr = --------------------------
%        (4 *pi * d)^2 * L
%  Pr (dB) = Pt + Gt + Gr + 20*log10(lambda/(4*pi*d)) - 10*log10(L) [2]
%
% [1] Wikipedia contributors. "Friis transmission equation." Wikipedia, The Free Encyclopedia. Wikipedia,
%     The Free Encyclopedia, 10 Jul. 2016. Web. 10 Jul. 2016. 
% [2] Mathuranathan, "Friis Free Space Propagation Model". 27. Sept. 2013. 
%
% Copyright (c) 2016 Hoa Van Nguyen
%
% This software is distributed under the GNU General Public 
%%
    Pr = Pt + Gt + Gr + 20*log10(Antenna_Gain'*lambda/(4*pi)) -20*log10(d) -L; % dB
    
end