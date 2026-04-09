use serde::{Deserialize, Serialize};

/// Represents the sparse spike outputs for a single 5ms hardware tick.
/// 1 = Spike Fired (Excitatory), 0 = No Spike
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TelemetrySpikes {
    pub power_spike: u8,
    pub pcie_rx_spike: u8,
    pub thermal_inhibit_spike: u8, // 1 = Hardware is stressed, suppress network
}

pub struct DerivativeEncoder {
    last_power_mw: u32,
    last_pcie_rx_kbps: u32,
    
    // Thresholds: How violent must the change be to trigger a spike?
    power_delta_threshold_mw: i32,
    pcie_delta_threshold_kbps: i32,
    thermal_limit_c: u32,
}

impl DerivativeEncoder {
    pub fn new(power_threshold_w: u32, pcie_threshold_mbps: u32, thermal_limit: u32) -> Self {
        Self {
            last_power_mw: 0,
            last_pcie_rx_kbps: 0,
            // Convert to matching units (mW and KB/s)
            power_delta_threshold_mw: (power_threshold_w * 1000) as i32,
            pcie_delta_threshold_kbps: (pcie_threshold_mbps * 1024) as i32,
            thermal_limit_c: thermal_limit,
        }
    }

    /// Processes a single 5ms sample from your NVML daemon and outputs binary spikes.
    pub fn encode_tick(
        &mut self, 
        current_power_mw: u32, 
        current_pcie_rx: u32, 
        current_temp_c: u32,
        throttle_reasons: u64
    ) -> TelemetrySpikes {
        
        // 1. Calculate the Deltas (The Derivative)
        let delta_power = (current_power_mw as i32) - (self.last_power_mw as i32);
        let delta_pcie = (current_pcie_rx as i32) - (self.last_pcie_rx_kbps as i32);

        // 2. Evaluate Excitatory Spikes (Sudden bursts of work)
        // We only care about positive spikes (increases in load) for excitation
        let power_spike = if delta_power > self.power_delta_threshold_mw { 1 } else { 0 };
        let pcie_rx_spike = if delta_pcie > self.pcie_delta_threshold_kbps { 1 } else { 0 };

        // 3. Evaluate Inhibitory Spikes (Hardware stress or limits)
        // If the GPU hits the thermal limit, or NVML reports a throttle reason, we fire an inhibitory spike.
        let thermal_inhibit_spike = if current_temp_c >= self.thermal_limit_c || throttle_reasons != 0 { 
            1 
        } else { 
            0 
        };

        // 4. Update memory for the next 5ms tick
        self.last_power_mw = current_power_mw;
        self.last_pcie_rx_kbps = current_pcie_rx;

        TelemetrySpikes {
            power_spike,
            pcie_rx_spike,
            thermal_inhibit_spike,
        }
    }

    /// Helper to process an entire Parquet batch (e.g., 2000 rows from DuckDB) into a spike train tensor.
    pub fn process_batch(
        &mut self, 
        power_array: &[u32], 
        pcie_array: &[u32], 
        temp_array: &[u32], 
        throttle_array: &[u64]
    ) -> Vec<TelemetrySpikes> {
        let mut spike_train = Vec::with_capacity(power_array.len());
        
        for i in 0..power_array.len() {
            spike_train.push(self.encode_tick(
                power_array[i], 
                pcie_array[i], 
                temp_array[i], 
                throttle_array[i]
            ));
        }
        
        spike_train
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivative_encoder_spikes() {
        // Thresholds: 50W sudden jump, 100MB/s sudden PCIe burst, 84C thermal limit
        let mut encoder = DerivativeEncoder::new(50, 100, 84);

        // Initial tick - no spikes expected because last values are 0
        let spikes = encoder.encode_tick(200_000, 10_000, 50, 0);
        assert_eq!(spikes.power_spike, 1); // 200W jump from 0
        assert_eq!(spikes.pcie_rx_spike, 0); // 10MB/s jump is less than 100MB/s (102400 KB/s)
        assert_eq!(spikes.thermal_inhibit_spike, 0);

        // Second tick - sudden jump in power
        // From 200W to 260W (60W jump > 50W threshold)
        let spikes = encoder.encode_tick(260_000, 10_000, 50, 0);
        assert_eq!(spikes.power_spike, 1);
        assert_eq!(spikes.pcie_rx_spike, 0);

        // Third tick - steady power, sudden PCIe burst
        // 10MB/s to 120MB/s (110MB jump > 100MB threshold)
        let spikes = encoder.encode_tick(260_000, 120 * 1024, 50, 0);
        assert_eq!(spikes.power_spike, 0);
        assert_eq!(spikes.pcie_rx_spike, 1);

        // Fourth tick - Thermal limit hit
        let spikes = encoder.encode_tick(260_000, 120 * 1024, 85, 0);
        assert_eq!(spikes.thermal_inhibit_spike, 1);

        // Fifth tick - Throttle reasons
        let spikes = encoder.encode_tick(260_000, 120 * 1024, 50, 1);
        assert_eq!(spikes.thermal_inhibit_spike, 1);
    }
}
