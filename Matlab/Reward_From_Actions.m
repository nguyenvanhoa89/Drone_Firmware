function R = Reward_From_Actions(UAV_Sets,pf,sys,obs, Ms, k, alpha,xh, Area)
    R = zeros(1,size(UAV_Sets,2));
    sigma_v = pf.sigma_v;
    nv =  size(sigma_v,1); 
%     sys_noise = mvnrnd(zeros(1,size(xh,1)),pf.sigma_u,pf.Ns)';
%     xk(:,:) = sys(k,  pf.particles(:,:,k), sys_noise);
%     xhk = sum(diag(pf.w(:,k)) * xk(:,:)')';
%     g_likelihood = zeros(1,Ms);
    gen_obs_noise =  mvnrnd(zeros(1,nv),sigma_v,Ms)';
    for a = 1:size(UAV_Sets,2)
%         R(a) = 1/norm(xh(:,k)-UAV_Sets(:,a));
%         sigma_v = abs(sigma_v - sigma_v*2/(norm(xh([1 2],k)-UAV_Sets([1 2],a)) + 1));
%         gen_obs_noise =  mvnrnd(zeros(1,nv),sigma_v,Ms)';
        if ~isempty(find(sign(UAV_Sets(:,a) - Area(:,1))==-1)) || ~isempty(find(sign(Area(:,2) - UAV_Sets(:,a)) ==-1))
            R(a) = -1e30;
        else
%             Future_Measurements = zeros(1,Ms);
%             gen_obs_noise =  mvnrnd(zeros(1,nv),sigma_v,Ms)';
            Future_Measurements = obs(k, xh(:,k), gen_obs_noise,UAV_Sets(:,a));
%             for j = 1:Ms
%                 Future_Measurements(j) = obs(k, xh(:,k), gen_obs_noise(),UAV_Sets(:,a));
%                 gamma(j) = 0;
%                 gamma_alpha(j) = 0;
%             end
            if mean(Future_Measurements) > pf.RSS_Threshold
                for j = 1:Ms
                    RSS_Sampled_std = Future_Measurements(j) - obs(k, pf.particles(:,:,k), 0,UAV_Sets(:,a));
%                     g_likelihood = 1/sqrt(2*pi *sigma_v^2) * exp(-0.5* RSS_Sampled_std.^2/sigma_v^2);
                    g_likelihood = mvnpdf(RSS_Sampled_std',zeros(1,nv),sigma_v);
                    gamma(j) = sum(diag(pf.w(:,k)) * g_likelihood);
                    gamma_alpha(j) = sum(diag(pf.w(:,k)) * (g_likelihood.^alpha));
%                     e_sq= sum( (diag(1./diag(sigma_v)*(repmat(z,[1 M])- Phi)).^2 );
%                     gz_vals= exp(-e_sq/2 - log(2*pi*prod(diag(model.D))));
%                     for i = 1:pf.Ns
%                         RSS_Sampled_std(i) = Future_Measurements(j) - obs(k, pf.particles(:,i,k), 0,UAV_Sets(:,a));
%                         g_likelihood(j,i) = 1/sqrt(2*pi *sigma_v^2) * exp(-0.5* RSS_Sampled_std(i)^2/sigma_v^2);
%     %                     g_likelihood(j,i) = pf.p_yk_given_xk(k, Future_Measurements(j), pf.particles(:,i,k),UAV_Sets(:,a));
%                         gamma(j) = gamma(j) + pf.w(i,k) * g_likelihood(j,i);
%                         gamma_alpha(j) = gamma_alpha(j) + pf.w(i,k) * g_likelihood(j,i)^alpha;
%                     end

                end
                R(a) = 0;
                for j = 1: Ms
                   R(a) = R(a) + 1/(alpha-1) * gamma(j) * log(abs(gamma_alpha(j))/abs(gamma(j))^alpha) ;
                end
            else
                R(a) = -1e30;
            end
                
        end
    end
end