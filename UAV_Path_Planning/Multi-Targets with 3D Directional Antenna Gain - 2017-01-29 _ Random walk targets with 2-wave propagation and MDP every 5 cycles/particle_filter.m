function [xhk, pf] = particle_filter(sys,obs, uavk, yk, pf, resampling_strategy)
%% Generic particle filter
%
% Note: when resampling is performed on each step this algorithm is called
% the Bootstrap particle filter
%
% Usage:
% [xhk, pf] = particle_filter(sys, yk, pf, resamping_strategy)
%
% Inputs:
% sys  = function handle to process equation
% yk   = observation vector at time k (column vector)
% pf   = structure with the following fields
%   .k                = iteration number
%   .Ns               = number of particles
%   .w                = weights   (Ns x T)
%   .particles        = particles (nx x Ns x T)
%   .gen_x0           = function handle of a procedure that samples from the initial pdf p_x0
%   .p_yk_given_xk    = function handle of the observation likelihood PDF p(y[k] | x[k])
%   .gen_sys_noise    = function handle of a procedure that generates system noise
% resampling_strategy = resampling strategy. Set it either to 
%                       'multinomial_resampling' or 'systematic_resampling'
%
% Outputs:
% xhk   = estimated state
% pf    = the same structure as in the input but updated at iteration k
%
% Reference:
% [1] Arulampalam et. al. (2002).  A tutorial on particle filters for 
%     online nonlinear/non-gaussian bayesian tracking. IEEE Transactions on 
%     Signal Processing. 50 (2). p 174--188

%% Programmed by:
% Diego Andres Alvarez Marin (diegotorquemada@gmail.com)
% Universidad Nacional de Colombia at Manizales, February 29, 2012

%%
k = pf.k;
if k == 1
   error('error: k must be an integer greater or equal than 2');
end

%% Initialize variables
Ns = pf.Ns;                              % number of particles
nx = size(pf.particles,1);               % number of states

wkm1 = pf.w(:, k-1);                     % weights of last iteration

if k == 2
%     RSS_Loss_Min = obs(1, [0 0 0]', 0 ,uavk);
%     d_Loss = norm(uavk);
%     d_Measured_Min = d_Loss*10^(abs(RSS_Loss_Min - yk )/20);
%     RSS_Loss_Max = obs(1, [0 0 0]', pf.sigma_v  ,uavk);
%     d_Measured_Max = d_Loss*10^(abs(RSS_Loss_Max - yk )/20);
   for i = 1:Ns                          % simulate initial particles
%       pf.particles(:,i,1) = randomDisc3(uavk, d_Measured_Min,d_Measured_Max,0); % at time k=1
      pf.particles(:,i,1) = pf.gen_x0();
   end   
   wkm1 = repmat(1/Ns, Ns, 1);           % all particles have the same weight
end

%%
% The importance sampling function:
% PRIOR: (this method is sensitive to outliers)   THIS IS THE ONE USED HERE
% q_xk_given_xkm1_yk = pf.p_xk_given_xkm1;

% OPTIMAL:
% q_xk_given_xkm1_yk = q_xk_given_xkm1^i_yk;
% Note this PDF can be approximated by MCMC methods: they are expensive but 
% they may be useful when non-iterative schemes fail

%% Separate memory
xkm1 = pf.particles(:,:,k-1); % extract particles from last iteration;
xk   = zeros(size(xkm1));     % = zeros(nx,Ns);
wk   = zeros(size(wkm1));     % = zeros(Ns,1);
% 

% d_Measured_Mean = (d_Measured_Min + d_Measured_Max)/2;
% dm1 = xkm1 - uavk;
% for i = 1:Ns
%    ddm1(i) = norm(dm1(:,i)) ;
% end
% mean_ddm1 = mean(ddm1);
% cov_ddm1 = cov(ddm1);
% sigma_u = d_Loss *[1 1 0];
% % pf.gen_sys_noise = @(u) mvnrnd(zeros(1,3),sigma_u,1)'; 
% sigma_v = 1;
%% Algorithm 3 of Ref [1]
if yk >= pf.RSS_Threshold
    sys_noise = mvnrnd(zeros(1,nx),pf.sigma_u,pf.Ns)';
    xk(:,:) = sys(k, xkm1(:,:), sys_noise);
    RSS_Sampled_std = yk - obs(k, xk(:,:), 0,uavk);
    sigma_v = pf.sigma_v;
    wk = diag(wkm1) * mvnpdf(RSS_Sampled_std',zeros(1,size(1,sigma_v)),sigma_v);
%     for i = 1:Ns
%        % xk(:,i) = sample_vector_from q_xk_given_xkm1_yk given xkm1(:,i) and yk
%        % Using the PRIOR PDF: pf.p_xk_given_xkm1: eq 62, Ref 1.
%     %    [adj(1), adj(2), adj(3)] = randomDisc3([0 0 0]', d_Measured_Min - mean_ddm1, d_Measured_Max  - mean_ddm1);
%        xk(:,i) = sys(k, xkm1(:,i), pf.gen_sys_noise()) ;
%     %    xk(:,i) = [randomDisc2(2, d_Measured_Min, d_Measured_Max);0];
%     %    xk(:,i) = randomDisc(uavk,3, d_Measured_Min, d_Measured_Max);
%     %     if norm(uavk) > d_Measured_Max
%     %         xk(:,i) = xkm1(:,i);
%     %     else
%     %          [xk(1,i),xk(2,i),xk(3,i)] = randomDisc3(uavk, d_Measured_Min, d_Measured_Max);
%     %     end
%      wk(i) = wkm1(i) * pf.p_yk_given_xk(k, yk, xk(:,i),uavk);
% 
%     %    d_Sample(i) = norm(xk(:,i)-uavk);
%     %    if d_Sample(i) > d_Measured_Min && d_Sample(i) < d_Measured_Max
%     %        wk(i) = wkm1(i) * pf.p_yk_given_xk(k, yk, xk(:,i),uavk);
%     %    else
%     %        wk(i) = 1e-18;
%     %    end
%        % Equation 48, Ref 1.
%        % wk(i) = wkm1(i) * p_yk_given_xk(yk, xk(:,i))*p_xk_given_xkm1(xk(:,i), xkm1(:,i))/q_xk_given_xkm1_yk(xk(:,i), xkm1(:,i), yk);
% 
%        % weights (when using the PRIOR pdf): eq 63, Ref 1
%     %     RSS_Sampled(i) = obs(1, xk(:,i), 0 ,uavk) ;
%     %     RSS_Sampled_std(i) = abs(yk - RSS_Sampled(i));
%     %     wk(i) = wkm1(i) / RSS_Sampled_std(i);
% 
%     %    likelihood(i) = 1/sqrt(2*pi *sigma_v^2) * exp(-0.5* RSS_Sampled_std(i)^2/sigma_v^2);
%     %    p_yk_given_xk1(i) = pf.p_yk_given_xk(k, yk, xk(:,i),uavk);
% 
%        % weights (when using the OPTIMAL pdf): eq 53, Ref 1
%        % wk(i) = wkm1(i) * p_yk_given_xkm1(yk, xkm1(:,i)); % we do not know this PDF
%     end

else
    xk = xkm1;
    wk = wkm1;
end
% RSS_Loss = obs(k, [0;0;0], 0,[0;0;0]);
% RSS_Noise = obs(k, [0;0;0], 1,[0;0;0]);
% d_Measured = 49*10^(abs(RSS_Loss - yk )/20);
% d_noise = 49*10^(abs(RSS_Loss - RSS_Noise )/20);
% if yk < -124
%    wk = wkm1; 
% end

%% Normalize weight vector
wk = wk./sum(wk);

%% Calculate effective sample size: eq 48, Ref 1
Neff = 1/sum(wk.^2);

%% Resampling
% remove this condition and sample on each iteration:
% [xk, wk] = resample(xk, wk, resampling_strategy);
%if you want to implement the bootstrap particle filter
resample_percentaje = 0.5;
Nt = resample_percentaje*Ns;
if Neff < Nt
   disp('Resampling ...')
   [xk, wk] = resample(xk, wk, resampling_strategy);
   % {xk, wk} is an approximate discrete representation of p(x_k | y_{1:k})
end

%% Compute estimated state
xhk = zeros(nx,1);
for i = 1:Ns
   xhk = xhk + wk(i)*xk(:,i);
end

%% Store new weights and particles
pf.w(:,k) = wk;
pf.particles(:,:,k) = xk;

return; % bye, bye!!!

%% Resampling function
function [xk, wk, idx] = resample(xk, wk, resampling_strategy)

Ns = length(wk);  % Ns = number of particles

% wk = wk./sum(wk); % normalize weight vector (already done)

switch resampling_strategy
   case 'multinomial_resampling'
      with_replacement = true;
      idx = randsample(1:Ns, Ns, with_replacement, wk);
%{
      THIS IS EQUIVALENT TO:
      edges = min([0 cumsum(wk)'],1); % protect against accumulated round-off
      edges(end) = 1;                 % get the upper edge exact
      % this works like the inverse of the empirical distribution and returns
      % the interval where the sample is to be found
      [~, idx] = histc(sort(rand(Ns,1)), edges);
%}
   case 'systematic_resampling'
      % this is performing latin hypercube sampling on wk
      edges = min([0 cumsum(wk)'],1); % protect against accumulated round-off
      edges(end) = 1;                 % get the upper edge exact
      u1 = rand/Ns;
      % this works like the inverse of the empirical distribution and returns
      % the interval where the sample is to be found
      [~, idx] = histc(u1:1/Ns:1, edges);
   % case 'regularized_pf'      TO BE IMPLEMENTED
   % case 'stratified_sampling' TO BE IMPLEMENTED
   % case 'residual_sampling'   TO BE IMPLEMENTED
    case 'SIR'
        idx = zeros(size(wk));
        N=length(wk);
        w=cumsum(wk);
        bin=1/N*rand(1)+1/N*(0:N-1);idxx=1;
        for t=1:N
            while bin(t)>=w(idxx)
                idxx=idxx+1;
            end
        idx(t) = idxx; 
        end
        
    case 'RPF'
        with_replacement = true;
        idx = randsample(1:Ns, Ns, with_replacement, wk);
        
        
    case 'soft_systematic_resampling'
        [idx, wk] = rs_soft_systematic(wk);
   otherwise
      error('Resampling strategy not implemented')
end;

                % extract new particles
switch resampling_strategy
       case 'soft_systematic_resampling'
            fprintf('soft_systematic_resampling');
             xk = xk(:,idx);    
       case 'RPF'
            fprintf('RPF');       
            S=xk * diag(wk) * xk'; %empirical covariance matrix 
            L=chol(S,'lower'); %the square root matrix of S 
            m =size(xk,1);
            epsilon=zeros(m,Ns); 
            A=(4/(m+2))^(1/(m+4)); 
            h=A*(Ns^(-1/(m+4))); 
            xk = xk(:,idx);     
            wk = repmat(1/Ns, 1, Ns); 
            for i=1:Ns 
                epsilon(:,i)=(h*L)*randn(1,m)'; 
                xk(:,i)=xk(:,i)+h.*(L*(epsilon(:,i))); 
            end
            
    otherwise
      xk = xk(:,idx);     
      wk = repmat(1/Ns, 1, Ns); 
end;
        
% wk = repmat(1/Ns, 1, Ns);          % now all particles have the same weight

return;  % bye, bye!!!

function [ndx, neww] = rs_soft_systematic(w, bratio, rratio)
% function [ndx, neww] = rs_soft_systematic(w, bratio, rratio)
%
% INPUT
% w = normalised weights,
% bratio = softness of resampling -- 0<bratio<=1
% rratio = proportion of stochastic resammpling -- rratio>1
%
% bratio controls the number of particles spawned by heavy particles
%  if bratio is small, the particles tend more to stay heavy rather than
%  spawning
%
% rratio controls the number of particles that are considered candidates for
%  resampling when eliminating the light particles
%
% OUTPUT
% ndx = resampled particle indexes
% neww = new weights
%
% Victoria University of Wellington
% Paul Teal,  8 August 2012
% modified to properly treat tails, P Teal, Thursday 5 September 2013
% modified P Choppala, 3 June 2014

% P Choppala, P Teal, M Frean, IEEE SSP Workshop, GoldCoast, AUS, 2014
% Soft systematic resampling
% Victoria University of Wellington, New Zealand


if nargin <2
  bratio = 0.9;    %   0<bratio<=1
end

if nargin <3
  rratio = 2.5;  %     rratio>1
end

N = length(w); % no. of particles

% Soft resampling
[val,ind] = sort(w,'descend');
tmp = max(1, floor(N*bratio*val));

cc=1;
for p1=1:N
  ndx(cc:cc+tmp(p1)-1) = ind(p1);
  neww(cc:cc+tmp(p1)-1) = val(p1)/tmp(p1);
  cc=cc+tmp(p1);
end
M=length(ndx);

% Soft systematic ressampling
if M>N
  Noverflow = M - N;
  Nsmall    = min(M,round(rratio*Noverflow));
  Nresamp   = Nsmall - (M-N);
  light_ndxes = M-Nsmall+1:M;
  resamp_ndxes = M-Nsmall+1:N;

  ws=neww(light_ndxes);
  neww2=sum(ws)/Nresamp;
  ws=ws/sum(ws);
  ndx3=rs_systematic(ws, Nresamp);
  ndx(resamp_ndxes) = ndx(light_ndxes(ndx3));
  neww(resamp_ndxes) = neww2;
end

ndx=ndx(1:N);
neww=neww(1:N);
return;


function [indx,neww]=rs_systematic(w, D)
% function [indx,neww]=rs_systematic(w, D)
% Systematic resampler
% Allows the possibility to draw fewer samples than the no. of weights
% Doucet et al., SMC methods in practice, 2001.
% Arulampalam et al., Tutorial paper, 2002.
%
% Victoria University of Wellington,
% P Teal, Tuesday 4 September 2013
% modified P Choppala, Tue 3 June 2014

N=length(w);

if nargin<2
  D = N;
end

Q=cumsum(w);
indx=zeros(1,D);

u=([0:N-1]+rand(1))/N;
j=1;

for i=1:D
  while (Q(j)<u(i))
    j=j+1;
  end
  indx(i)=j;
end

neww=ones(1,N)/N;
return;
