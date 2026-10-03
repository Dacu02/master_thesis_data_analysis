# master_thesis_data_analysis
Repository containing script for analzying the data collected during the master thesis.

## [Extraction](./extraction/)
This folder contains the script [`computeDatabase.m`](./extraction/computeDatabase.m) which reads the data from the `.bag` files and extracts the strokes returned from passing the content to ideLog3D. The result is a new folder structure containing the extracted strokes and the full reconstructed trajectory in two `.csv` files.

### Extraction pipeline
Before passing the data to ideLog3D, the [`computeDatabase.m`](./extraction/computeDatabase.m) script performs the following steps:
1. Reads the `.bag` files and extracts the raw data recorded from the hardware device $r_t$
2. The signal is resampled to a fixed frequency of 500 Hz $r_{t_k}$
3. The resampled signal is trimmed from boundary noise and outliers $t_{t_k}$
4. The trimmed signal boundaries are padded with a quartic ramp that goes to zero in order to avoid errors from the ideLog3D algorithm $p_{t_k}$
5. The padded signal is filtered with a Chebyshev type II low-pass filter with a cutoff frequency of 16Hz and a stopband attenuation of 80db $f_{t_k}$
6. The filtered signal is passed to ideLog3D which returns the reconstructed trajectory and the extracted strokes $s_{t_k}$