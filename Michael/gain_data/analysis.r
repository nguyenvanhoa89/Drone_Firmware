library(ggplot2)
library(dplyr)

# Read in raw gain data
gain_data <- read.csv('processed_pulse_data.csv')
gain_data$gain <- gain_data$lna_gain + gain_data$vga_gain
gain_data$rssi_norm <- gain_data$rssi / median(with(subset(gain_data, gain == 0), rssi))
gain_data$rssi_db = 10 * log10(gain_data$rssi_norm)

# Calculate linear regression
regression = with(subset(gain_data, rssi < 0.9), lm(rssi_db ~ 0 + gain))
gain_model <- coef(regression)

# A basic scatter plot in log scale, to display the distributions
svg('gain-data-plot-1.svg', width = 6, height = 4);
ggplot(gain_data, aes(gain, rssi_db)) + geom_point() +
    geom_abline(intercept = 0, slope = gain_model, color = "red") +
    xlab("Gain setting (dB)") + ylab("Actual gain (dB)") + ggtitle("Actual gain vs Selected gain") +
    theme_bw()
dev.off()

# A plot of averages
svg('gain-data-plot-2.svg', width = 6, height = 4);
gain_data_avg <- summarise(group_by(gain_data, gain), rssi_db = 10 * log10(mean(rssi_norm)))
ggplot(gain_data_avg, aes(gain, rssi_db)) + geom_point() + 
    geom_abline(intercept = 0, slope = gain_model) + theme_bw()
dev.off()

ggplot() + geom_abline(intercept = gain_model[1], slope = gain_model[2])

# A plot of all values where the gain is set separately
ggplot(subset(gain_data, vga_gain == 0), aes(lna_gain, rssi_db)) + geom_point() +
    geom_point(data = subset(gain_data, lna_gain == 0), aes(vga_gain, rssi_db), color = "Red") + theme_classic()