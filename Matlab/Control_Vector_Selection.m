function [uavkp1, Reward] = Control_Vector_Selection (k, pf,sys,obs, UAV_Sets, Ms, alpha, Area)
    Ns = pf.Ns;
    nx = size(pf.particles,1);              % number of states
    
    
    sigma_v = pf.sigma_v;
    nv =  size(sigma_v,1); 
%     if k == 2 % Intialize particle when k = 1
%         pf.particles(:,:,k-1) = pf.gen_x0;
%         xkm1 = pf.particles(:,:,k-1);
%     else
%         xkm1 = pf.particles(:,:,k-1);
%     end 
    xk = pf.particles(:,:,k);
    sys_noise = mvnrnd(zeros(1,nx),pf.sigma_u,pf.Ns)';
%     gen_obs_noise =  mvnrnd(zeros(1,nv),sigma_v,1)';
    xkp1 = sys(k, xk(:,:), sys_noise);      % Next k+1 location
    xhkp1 = sum(diag(pf.w(:,k)) * xkp1')';
    gain = Get_Antenna_Gain( xhkp1, UAV_Sets, pf.gain_angle);
    UAV_Sets = UAV_Sets(:,gain>=prctile(gain,80)); % Get new control action set with good gain only. 80 is the best
    nu = size(UAV_Sets,2);
    R = zeros(1,nu);                        % Reward function
    for a = 1:nu
        u = UAV_Sets(:,a);
        
        if ~isempty(find(sign(u(1:2) - Area(1:2,1))==-1)) || ~isempty(find(sign(Area(1:2,2) - u(1:2)) ==-1))
            R(a) = -1e30;
        else
            for m = 1:Ms
                jm = randi(Ns);
                ykp1(m) = obs(k+1, xkp1(:,jm), mvnrnd(zeros(1,nv),sigma_v,1)',u, pf.gain_angle);
%                 ykp1(m) = obs(k+1, xhkp1, mvnrnd(zeros(1,nv),sigma_v,1)',u);
                RSS_Sampled_std = ykp1(m) - obs(k+1, xkp1, 0,u, pf.gain_angle);
                % For real drone test only
%                 RSS_Sampled_std = RSS_Sampled_std - mean(RSS_Sampled_std);
%                 sigma_v = std(RSS_Sampled_std);
%                 sigma_v = sigma_v.^2;
                %%%
                g_likelihood = mvnpdf(RSS_Sampled_std',zeros(1,nv),sigma_v);
                wkp1 = g_likelihood./sum(g_likelihood);
                gamma_alpha(m) =  1/(alpha-1) * log(Ns^(alpha-1) * sum(wkp1.^alpha));
            end
            R(a) = mean(gamma_alpha);
        end
    end
    [Reward,best_u] = max(R,[],2);
    uavkp1 = UAV_Sets(:,best_u);
end