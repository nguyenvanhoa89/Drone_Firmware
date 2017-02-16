use common::hackrf::{HackRFConfig, GainConfig};
use common::signal::Pulse;

pub struct GainControl {
    current_gain: GainConfig,
    wait_time: u64,
    samples_since_pulse: u64,
}

impl GainControl {
    pub fn new(config: &HackRFConfig) -> GainControl {
        GainControl {
            current_gain: GainConfig { lna_gain: config.lna_gain, vga_gain: config.vga_gain },
            wait_time: 2 * config.samp_rate as u64,
            samples_since_pulse: 0,
        }
    }

    pub fn update_sample_count(&mut self, elapsed_samples: u64) {
        self.samples_since_pulse += elapsed_samples;
    }

    pub fn check_gain(&mut self, pulses: &[Pulse]) -> Option<GainConfig> {
		println!("current gain: {:?}",self.current_gain);
		//let max_gain = GainConfig {lna_gain: 40, vga_gain: 62};
		let default_gain = GainConfig {lna_gain: 40, vga_gain: 32};
		self.current_gain = default_gain;
		return Some(self.current_gain );
		/*
        let max_pulse = pulses.iter().map(|x| x.signal_strength)
            .fold(None, |old, x| Some(x.max(old.unwrap_or(x))));
		println!("Max pulse: {:?}",max_pulse);
        if let Some(pulse) = max_pulse {
            self.samples_since_pulse = 0;
	    println!("pulse: {:?}",pulse);
            if pulse > 0.9{
                self.current_gain = decrease_gain(self.current_gain.total_gain(), pulse);
                return Some(self.current_gain);
            }
	
        }
        println!("sample since pulse: {:?}",self.samples_since_pulse);
			//println!("wait time: {:?}",self.wait_time);
			
	        if self.samples_since_pulse > self.wait_time {
	            self.samples_since_pulse -= self.wait_time;
	            self.current_gain = increase_gain(self.current_gain.total_gain());
	
	            return Some(self.current_gain);
	        }
		
        None
        */

//	return Some(GainConfig::new(80))

    }
}

pub fn increase_gain(old_gain: u32) -> GainConfig {
    GainConfig::new(old_gain + 4)
}

pub fn decrease_gain(old_gain: u32, _max_value: f32) -> GainConfig {
    if old_gain > 4 {
        GainConfig::new(old_gain - 4)
    }
    else {
        GainConfig::new(0)
    }
}
