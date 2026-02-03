use crate::consts::{CLIP, EPS, FEATURE_SIZE, ROLLING_WINDOW, SKIPPED_TIMESTEPS, THRESHOLD};

/// Generate classification targets from closing prices.
///
/// - `closes`: slice of close prices.
/// - `horizon`: number of days ahead to predict (1 for next-day).
///
/// Returns a vector of length `closes.len() - horizon` where:
/// - 1 means price up.
/// - 0 means price down.
pub fn generate_targets(closes: &[f32], horizon: usize) -> Vec<i32> {
    let n = closes.len();
    assert!(horizon > 0 && n > horizon, "Invalid horizon or data length");

    let mut targets = Vec::with_capacity(n - horizon);

    for t in 0..(n - horizon) {
        let curr = closes[t].max(EPS);
        let fut = closes[t + horizon].max(EPS);

        let ret = (fut / curr).ln();

        let target = if ret > THRESHOLD { 1 } else { 0 };

        targets.push(target);
    }

    targets
}

/// Normalize the output from [process] into rolling mean/std and clipped data.
pub fn normalize(features: &[[f32; FEATURE_SIZE]]) -> Vec<[f32; FEATURE_SIZE]> {
    let n = features.len();
    let mut out = vec![[0.0; FEATURE_SIZE]; n];

    for f_idx in 0..FEATURE_SIZE {
        // Compute rolling mean/std
        for t in 0..n {
            if t < ROLLING_WINDOW {
                // Not enough history: leave as 0 or just normalize by what is available
                out[t][f_idx] = 0.0;
            } else {
                let slice = &features[t - ROLLING_WINDOW..t];
                let values: Vec<f32> = slice.iter().map(|row| row[f_idx]).collect();

                let mean = values.iter().sum::<f32>() / ROLLING_WINDOW as f32;
                let var =
                    values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / ROLLING_WINDOW as f32;
                let std = var.sqrt().max(EPS);

                // causal z-score + clip
                let z = (features[t][f_idx] - mean) / std;
                out[t][f_idx] = z.clamp(-CLIP, CLIP);
            }
        }
    }

    out
}

/// Processes the input data into features:
/// - log_return_1
/// - log_return_3
/// - log_return_5
/// - log_return_10
/// - rolling_std_5
/// - rolling_std_10
/// - rolling_std_20
/// - sma_5_ratio
/// - sma_10_ratio
/// - ema_12_ratio
/// - ema_26_ratio
/// - candle_body
/// - upper_wick
/// - lower_wick
/// - candle_range
/// - volume_ratio
/// - volume_change
pub fn process(
    opens: &[f32],
    closes: &[f32],
    volumes: &[f32],
    highs: &[f32],
    lows: &[f32],
) -> Vec<[f32; FEATURE_SIZE]> {
    let n = closes.len();

    assert!(
        opens.len() == n && volumes.len() == n && highs.len() == n && lows.len() == n,
        "Input vectors must have the same length"
    );

    assert!(
        n > SKIPPED_TIMESTEPS,
        "Not enough timesteps: need {}, got {}",
        SKIPPED_TIMESTEPS,
        n
    );

    let lr1 = log_return(closes, 1);
    let lr3 = log_return(closes, 3);
    let lr5 = log_return(closes, 5);
    let lr10 = log_return(closes, 10);

    let vol5 = rolling_std(&lr1, 5);
    let vol10 = rolling_std(&lr1, 10);
    let vol20 = rolling_std(&lr1, 20);

    let sma5 = sma_ratio(closes, 5);
    let sma10 = sma_ratio(closes, 10);
    let ema12 = ema_ratio(closes, 12);
    let ema26 = ema_ratio(closes, 26);

    let (body, upper, lower, range) = candle_features(opens, closes, highs, lows);

    let vol_ratio = volume_ratio(volumes, 20);
    let vol_change = volume_change(volumes);

    let mut out = vec![[0.0; FEATURE_SIZE]; n];

    for i in 0..n {
        out[i] = [
            lr1[i],
            lr3[i],
            lr5[i],
            lr10[i],
            vol5[i],
            vol10[i],
            vol20[i],
            sma5[i],
            sma10[i],
            ema12[i],
            ema26[i],
            body[i],
            upper[i],
            lower[i],
            range[i],
            vol_ratio[i],
            vol_change[i],
        ];
    }

    out
}

/// Compute k-day log returns for a slice of prices.
///
/// The first `k` entries are set to 0.0.
fn log_return(prices: &[f32], k: usize) -> Vec<f32> {
    let n = prices.len();
    let mut out = vec![0.0; n];

    for i in k..n {
        let prev = prices[i - k].max(EPS);
        let curr = prices[i].max(EPS);
        out[i] = (curr / prev).ln();
    }

    out
}

/// Compute rolling standard deviation for a slice of values with a given window size.
///
/// The first `window` entries are set to 0.0.
fn rolling_std(values: &[f32], window: usize) -> Vec<f32> {
    let n = values.len();
    let mut out = vec![0.0; n];

    for i in window..n {
        let slice = &values[i - window..i];
        let mean = slice.iter().sum::<f32>() / window as f32;
        let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / window as f32;

        out[i] = var.sqrt();
    }

    out
}

/// Compute the Simple Moving Average (SMA) over a given window size.
///
/// The first `window` entries are set to 0.0.
fn sma_ratio(values: &[f32], window: usize) -> Vec<f32> {
    let n = values.len();
    let mut out = vec![0.0; n];

    for i in window..n {
        let slice = &values[i - window..i];
        let mean = slice.iter().sum::<f32>() / window as f32;
        out[i] = values[i].max(EPS) / mean;
    }

    out
}

/// Compute the Exponential Moving Average (EMA) over a given window size.
fn ema_ratio(values: &[f32], window: usize) -> Vec<f32> {
    let n = values.len();
    let mut ema = vec![0.0; n];

    let alpha = 2.0 / (window as f32 + 1.0);
    ema[0] = values[0];

    for i in 1..n {
        ema[i] = alpha * values[i] + (1.0 - alpha) * ema[i - 1];
    }

    ema.iter()
        .zip(values)
        .map(|(e, v)| e / v.max(EPS))
        .collect()
}

/// Computes candle features:
/// - candle_body: (close - open) / close
/// - upper_wick: (high - max(open, close)) / close
/// - lower_wick: (min(open, close) - low) / close
/// - candle_range: (high - low) / close
fn candle_features(
    opens: &[f32],
    closes: &[f32],
    highs: &[f32],
    lows: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = closes.len();

    let mut body = vec![0.0; n];
    let mut upper = vec![0.0; n];
    let mut lower = vec![0.0; n];
    let mut range = vec![0.0; n];

    for i in 0..n {
        let c = closes[i].max(EPS);
        body[i] = (closes[i] - opens[i]) / c;
        upper[i] = (highs[i] - opens[i].max(closes[i])) / c;
        lower[i] = (opens[i].min(closes[i]) - lows[i]) / c;
        range[i] = (highs[i] - lows[i]) / c;
    }

    (body, upper, lower, range)
}

/// Compute the volume ratio: volume / rolling mean over a given window size.
///
/// The first `window` entries are set to 0.0.
fn volume_ratio(volumes: &[f32], window: usize) -> Vec<f32> {
    let n = volumes.len();
    let mut out = vec![0.0; n];

    for i in window..n {
        let slice = &volumes[i - window..i];
        let mean = slice.iter().sum::<f32>() / window as f32;
        out[i] = volumes[i] / mean.max(EPS);
    }

    out
}

/// Compute day-over-day log volume change: `log(volume[t] / volume[t-1])`
///
/// The first entry is set to 0.0.
fn volume_change(volumes: &[f32]) -> Vec<f32> {
    let n = volumes.len();
    let mut out = vec![0.0; n];

    for i in 1..n {
        out[i] = (volumes[i] / volumes[i - 1].max(EPS)).ln();
    }

    out
}
