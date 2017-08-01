foundIndex = [est.foundIndex{:}]';
foundIndex = sort(foundIndex,1);
est.RMS_Result = zeros(size(foundIndex,1),ntarget);
est.detCovResult  = zeros(size(foundIndex,1),ntarget);
for i=1:ntarget
    currentIndex = foundIndex(foundIndex <= est.foundIndex{i});
    est.RMS_Result(1:size(currentIndex,1),i) = est.RMS{i}(currentIndex)';
    for j=1:size(currentIndex,1)
         est.detCovResult(j,i) = det(cov(est.pf{i}.particles(1:2,:,currentIndex(j))'));
    end
%     detCovResult(1:size(currentIndex,1),i) = det(cov(est.pf{i}.particles(1:2,:,currentIndex')'));
end
