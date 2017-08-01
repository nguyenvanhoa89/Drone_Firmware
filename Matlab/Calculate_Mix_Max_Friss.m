function [fmin, fmax] =  Calculate_Mix_Max_Friss(Pt,Gt, Gr, L, R_max, uav_height, loc_err, phi)
% R_max = 200;
% PtW = 0.5e-3;
% Pt = 10*log10(PtW); %dBm
f = 173e6;
c = physconst('lightspeed');
lambda = c/f;
% Gt = 0;   %dBm
% Gr = -15; %dBm
% L = 12; %dBm  
fun = @(x) Pt + Gt + Gr -L + 10*x(7)*log10(lambda/(4*pi)) -5*x(7)*log10((x(1) - x(4))^2 + (x(2) - x(5))^2 + (x(6))^2)  ;  
%% Friis equation notation:
% x_target = x(1:3). x(3) = 0
% x_uav = x(4:6)
% propagation loss = x(7)
%%
fun_max = @(x)-fun(x);
lb = [zeros(1,5) uav_height - loc_err phi(1) ];
ub = [R_max * ones(1,5) uav_height + loc_err phi(2)];
x0 = [ones(1,5) uav_height mean(phi)];

A = [];
b = [];
Aeq = [];
beq = [];
[~,fmin] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
[~,fmaxr] = fmincon(fun_max,x0,A,b,Aeq,beq,lb,ub);
fmax = -fmaxr;
end