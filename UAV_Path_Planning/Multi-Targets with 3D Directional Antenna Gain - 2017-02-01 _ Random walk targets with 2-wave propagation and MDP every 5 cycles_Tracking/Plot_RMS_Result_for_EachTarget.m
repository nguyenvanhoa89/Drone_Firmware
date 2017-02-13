figure(3);
for i = 1:ntarget
    subplot(1,ntarget,i);
    plot(1:k, est.RMS{i}(1,1:k),'-k');
    title(['RMS of target ',num2str(i)]);
    ylabel('RMS (m)');
    xlabel('Cycle');
    grid on;
end