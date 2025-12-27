use crate::data::FEATURE_SIZE;

/// Generate log return targets for a single stock
///
/// # Arguments
/// - `closes`: slice of adjusted close prices (time-ordered)
/// - `horizon`: number of timesteps ahead to predict
///
/// # Returns
/// - Vec<f32> of log return targets, aligned with input slice.
///   The last `horizon` elements will be ignored (no future data)
pub fn generate_targets(closes: &[f32], horizon: usize) -> Vec<f32> {
    let n = closes.len();
    if horizon == 0 || n <= horizon {
        return vec![];
    }

    let mut targets = Vec::with_capacity(n - horizon);
    for i in 0..(n - horizon) {
        let log_return = (closes[i + horizon] / closes[i]).ln();
        targets.push(log_return);
    }

    targets
}

/// Processes the input data into:
/// -  log_return
/// -  volume_change
/// -  high_low_range
/// -  open_close_return
/// -  rolling_volatility_10
/// -  sma_distance_10
/// -  ema_distance_20
/// -  momentum_10
/// -  atr_14
/// -  bollinger_band_distance_20
///
/// and stores them in a vector of 10-float arrays.
pub fn process(
    opens: Vec<f32>,
    closes: Vec<f32>,
    volumes: Vec<f32>,
    highs: Vec<f32>,
    lows: Vec<f32>,
) -> Vec<[f32; FEATURE_SIZE]> {
    let n = closes.len();
    assert!(
        opens.len() == n && volumes.len() == n && highs.len() == n && lows.len() == n,
        "Input vectors must have the same length"
    );

    let mut out = vec![[0.0; FEATURE_SIZE]; n];

    let mut log_returns = vec![0.0; n];
    let mut tr = vec![0.0; n];

    // ---- Base features ----
    for i in 1..n {
        log_returns[i] = (closes[i] / closes[i - 1]).ln();
        out[i][0] = log_returns[i]; // log return

        out[i][1] = (volumes[i] - volumes[i - 1]) / volumes[i - 1]; // volume change
        out[i][2] = (highs[i] - lows[i]) / closes[i]; // high-low range
        out[i][3] = (closes[i] - opens[i]) / opens[i]; // open-close return

        // True Range (for ATR)
        tr[i] = (highs[i] - lows[i])
            .max((highs[i] - closes[i - 1]).abs())
            .max((lows[i] - closes[i - 1]).abs());
    }

    // ---- Rolling volatility (10) ----
    for i in 9..n {
        let mean = log_returns[i - 9..=i].iter().sum::<f32>() / 10.0;
        let var = log_returns[i - 9..=i]
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / 10.0;
        out[i][4] = var.sqrt();
    }

    // ---- SMA distance (10) ----
    for i in 9..n {
        let sma = closes[i - 9..=i].iter().sum::<f32>() / 10.0;
        out[i][5] = (closes[i] - sma) / sma;
    }

    // ---- EMA distance (20) ----
    let alpha = 2.0 / (20.0 + 1.0);
    let mut ema = closes[0];
    for i in 1..n {
        ema = alpha * closes[i] + (1.0 - alpha) * ema;
        if i >= 19 {
            out[i][6] = (closes[i] - ema) / ema;
        }
    }

    // ---- Momentum (10) ----
    for i in 10..n {
        out[i][7] = closes[i] / closes[i - 10] - 1.0;
    }

    // ---- ATR (14) ----
    for i in 13..n {
        let atr = tr[i - 13..=i].iter().sum::<f32>() / 14.0;
        out[i][8] = atr;
    }

    // ---- Bollinger Band distance (20) ----
    for i in 19..n {
        let window = &closes[i - 19..=i];
        let mean = window.iter().sum::<f32>() / 20.0;
        let std = (window.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 20.0).sqrt();

        let band_width = 2.0 * std;
        if band_width > 0.0 {
            out[i][9] = (closes[i] - mean) / band_width;
        }
    }

    out
}

/// Normalize the output from [process].
///
/// - log_return, volume_change, high_low_range, open_close_return,
///   SMA distance, EMA distance, momentum -> rolling z-score
/// - rolling_volatility, ATR -> divide by long-term mean
/// - Bollinger distance -> leave as-is
///
/// # Arguments
/// `window_z` - rolling window for z-score normalization (e.g. 60)
///
/// `window_scale` - long-term mean window for volatility/ATR (e.g. 252)
///
/// `clip` - clip z-score to +/- clip
pub fn normalize(
    features: &Vec<[f32; FEATURE_SIZE]>,
    window_z: usize,
    window_scale: usize,
    clip: f32,
) -> Vec<[f32; FEATURE_SIZE]> {
    let n = features.len();
    let mut out = vec![[0.0; FEATURE_SIZE]; n];

    // Extract per-feature vectors
    let mut log_return: Vec<f32> = features.iter().map(|x| x[0]).collect();
    let mut volume_change: Vec<f32> = features.iter().map(|x| x[1]).collect();
    let mut hl_range: Vec<f32> = features.iter().map(|x| x[2]).collect();
    let mut oc_return: Vec<f32> = features.iter().map(|x| x[3]).collect();
    let mut rolling_vol: Vec<f32> = features.iter().map(|x| x[4]).collect();
    let mut sma_dist: Vec<f32> = features.iter().map(|x| x[5]).collect();
    let mut ema_dist: Vec<f32> = features.iter().map(|x| x[6]).collect();
    let mut momentum: Vec<f32> = features.iter().map(|x| x[7]).collect();
    let mut atr: Vec<f32> = features.iter().map(|x| x[8]).collect();
    let bollinger: Vec<f32> = features.iter().map(|x| x[9]).collect(); // leave as-is

    // Helper for rolling z-score with clipping
    fn rolling_zscore(values: &[f32], window: usize, clip: f32) -> Vec<f32> {
        let n = values.len();
        let mut out = vec![0.0; n];
        for i in window..n {
            let slice = &values[i - window..i];
            let mean = slice.iter().sum::<f32>() / window as f32;
            let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / window as f32;
            let std = var.sqrt().max(1e-6);
            out[i] = ((values[i] - mean) / std).clamp(-clip, clip);
        }
        out
    }

    // Helper for dividing by long-term mean
    fn divide_by_long_mean(values: &[f32], window: usize) -> Vec<f32> {
        let n = values.len();
        let mut out = vec![0.0; n];
        for i in window..n {
            let mean = values[i - window..i].iter().sum::<f32>() / window as f32;
            out[i] = values[i] / mean.max(1e-6);
        }
        out
    }

    // Normalize rolling z-score features
    log_return = rolling_zscore(&log_return, window_z, clip);
    volume_change = rolling_zscore(&volume_change, window_z, clip);
    hl_range = rolling_zscore(&hl_range, window_z, clip);
    oc_return = rolling_zscore(&oc_return, window_z, clip);
    sma_dist = rolling_zscore(&sma_dist, window_z, clip);
    ema_dist = rolling_zscore(&ema_dist, window_z, clip);
    momentum = rolling_zscore(&momentum, window_z, clip);

    // Normalize volatility / ATR
    rolling_vol = divide_by_long_mean(&rolling_vol, window_scale);
    atr = divide_by_long_mean(&atr, window_scale);

    // Combine back into output
    for i in 0..n {
        out[i][0] = log_return[i];
        out[i][1] = volume_change[i];
        out[i][2] = hl_range[i];
        out[i][3] = oc_return[i];
        out[i][4] = rolling_vol[i];
        out[i][5] = sma_dist[i];
        out[i][6] = ema_dist[i];
        out[i][7] = momentum[i];
        out[i][8] = atr[i];
        out[i][9] = bollinger[i]; // unchanged
    }

    out
}
