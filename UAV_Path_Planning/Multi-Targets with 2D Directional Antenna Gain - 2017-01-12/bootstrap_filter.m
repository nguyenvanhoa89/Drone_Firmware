function [xhk, pf] = bootstrap_filter (k, pf,sys,obs, yk, uavk)
    Ns = pf.Ns;
    nx = size(pf.particles,1);               % number of states
    if k == 2 % Intialize particle when k = 1
        pf.particles(:,:,k-1) = pf.gen_x0;
        xkm1 = pf.particles(:,:,k-1);
        wkm1 = repmat(1/Ns, Ns, 1); 
    else
        wkm1 = pf.w(:, k-1);  
        xkm1 = pf.particles(:,:,k-1);
    end    
    if yk >= pf.RSS_Threshold
        sys_noise = mvnrnd(zeros(1,nx),pf.sigma_u,pf.Ns)';
        xk(:,:) = sys(k, xkm1(:,:), sys_noise);
        RSS_Sampled_std = yk - obs(k, xk(:,:), 0,uavk);
        sigma_v = pf.sigma_v;
        wk = diag(wkm1) * mvnpdf(RSS_Sampled_std',zeros(1,size(1,sigma_v)),sigma_v);
%         RSS_std_max = -103.15;
%         RSS_std_min = -124.98;
%         g_likelihhood = cdf('Normal',RSS_Sampled_std,RSS_std_min,sigma_v,'upper') - cdf('Normal',RSS_Sampled_std,RSS_std_max,sigma_v,'upper') ;
%         wk = diag(wkm1) * g_likelihhood';
       
    else
        xk = xkm1;
        wk = wkm1;
    end
    wk = wk./sum(wk);
    %% Random injection to escape local minimum
    p_Inject = 0.1; 
    if rand < p_Inject
        InjectNumber = round(p_Inject * pf.Ns);
        InjectLocation = randi(pf.Ns,InjectNumber,1);
        xk(:,InjectLocation) = [pf.R_max * rand(InjectNumber,1) pf.R_max * rand(InjectNumber,1) zeros(InjectNumber,1)]';
    end
    %% ---resampling
    idx= randsample(length(wk),Ns,true,wk);
    wk= ones(Ns,1)/Ns;
    xk= xk(:,idx);
    
    %% Compute estimated state
    xhk = sum(diag(wk) * xk');
    %% Store new weights and particles
    pf.w(:,k) = wk;
    pf.particles(:,:,k) = xk;
end