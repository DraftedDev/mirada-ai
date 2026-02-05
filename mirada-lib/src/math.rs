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
/// - volume_ema_ratio_12_26
/// - volume_spike_14
/// - average_true_range_14
/// - average_directional_index_14
/// - on_balance_volume
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
    let vol_ema_rat = volume_ema_ratio(volumes, 12, 26);
    let vol_spike = volume_spike(volumes, 14);

    let atr14 = atr(highs, lows, closes, 14);
    let adx14 = adx(highs, lows, closes, 14);
    let obv = obv(closes, volumes);

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
            vol_ema_rat[i],
            vol_spike[i],
            atr14[i],
            adx14[i],
            obv[i],
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

/// Computes the rolling mean over a given window size.
///
/// The first `window` timesteps are set to 0.0.
///
/// This is a causal rolling mean: the value at index `t` is the mean of the previous `window` values.
pub fn rolling_mean(values: &[f32], window: usize) -> Vec<f32> {
    let n = values.len();
    let mut out = vec![0.0; n];

    if window == 0 || n == 0 {
        return out;
    }

    // Sum of the first window elements
    let mut sum: f32 = values.iter().take(window.min(n)).sum();

    for i in 0..n {
        if i + 1 >= window {
            // rolling mean over window
            out[i] = sum / window as f32;

            // subtract the value leaving the window (if any)
            if i + 1 < n {
                sum -= values[i + 1 - window];
                // add the new value coming into the window
                sum += values[i + 1];
            }
        } else {
            // not enough history yet
            out[i] = 0.0;
        }
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

/// Computes the Average True Range (ATR), a measure of volatility:
///
/// ```text
/// ATR_t = rolling_mean(max(
///     High_t - Low_t,
///     |High_t - Close_{t-1}|,
///     |Low_t - Close_{t-1}|
/// ), window)
/// ```
///
/// The first `window` timesteps are set to 0.0.
fn atr(highs: &[f32], lows: &[f32], closes: &[f32], window: usize) -> Vec<f32> {
    let n = highs.len();
    let mut tr = vec![0.0; n];
    for i in 1..n {
        tr[i] = (highs[i] - lows[i])
            .max((highs[i] - closes[i - 1]).abs())
            .max((lows[i] - closes[i - 1]).abs());
    }
    rolling_mean(&tr, window) // or simple rolling mean instead of std
}

/// Computes the Average Directional Index (ADX), which measures trend strength:
///
/// ```text
/// +DM_t = High_t - High_{t-1} if it's greater than |Low_t - Low_{t-1}| else 0
/// -DM_t = Low_{t-1} - Low_t if it's greater than |High_t - High_{t-1}| else 0
/// DX_t = |+DI_t - -DI_t| / (+DI_t + -DI_t)
/// ADX_t = rolling_mean(DX, window)
/// ```
/// The first `window` timesteps are set to 0.0.
pub fn adx(highs: &[f32], lows: &[f32], closes: &[f32], window: usize) -> Vec<f32> {
    let n = highs.len();
    let mut plus_dm = vec![0.0; n];
    let mut minus_dm = vec![0.0; n];
    let mut tr = vec![0.0; n];

    for i in 1..n {
        let up = highs[i] - highs[i - 1];
        let down = lows[i - 1] - lows[i];

        plus_dm[i] = if up > down && up > 0.0 { up } else { 0.0 };
        minus_dm[i] = if down > up && down > 0.0 { down } else { 0.0 };

        tr[i] = (highs[i] - lows[i])
            .max((highs[i] - closes[i - 1]).abs())
            .max((lows[i] - closes[i - 1]).abs());
    }

    let mut smoothed_tr = vec![0.0; n];
    let mut smoothed_plus_dm = vec![0.0; n];
    let mut smoothed_minus_dm = vec![0.0; n];

    // initialize with simple sum over first `window` period
    let sum_tr = tr[1..=window].iter().sum::<f32>();
    let sum_plus = plus_dm[1..=window].iter().sum::<f32>();
    let sum_minus = minus_dm[1..=window].iter().sum::<f32>();

    smoothed_tr[window] = sum_tr;
    smoothed_plus_dm[window] = sum_plus;
    smoothed_minus_dm[window] = sum_minus;

    for i in (window + 1)..n {
        smoothed_tr[i] = smoothed_tr[i - 1] - (smoothed_tr[i - 1] / window as f32) + tr[i];
        smoothed_plus_dm[i] =
            smoothed_plus_dm[i - 1] - (smoothed_plus_dm[i - 1] / window as f32) + plus_dm[i];
        smoothed_minus_dm[i] =
            smoothed_minus_dm[i - 1] - (smoothed_minus_dm[i - 1] / window as f32) + minus_dm[i];
    }

    let mut plus_di = vec![0.0; n];
    let mut minus_di = vec![0.0; n];

    for i in window..n {
        if smoothed_tr[i] > 0.0 {
            plus_di[i] = smoothed_plus_dm[i] / smoothed_tr[i];
            minus_di[i] = smoothed_minus_dm[i] / smoothed_tr[i];
        }
    }

    let mut dx = vec![0.0; n];
    for i in window..n {
        let sum = plus_di[i] + minus_di[i];
        if sum != 0.0 {
            dx[i] = (plus_di[i] - minus_di[i]).abs() / sum;
        }
    }

    let mut adx = vec![0.0; n];
    adx[window] = dx[window..=window + window - 1].iter().sum::<f32>() / window as f32; // initial ADX

    for i in (window + 1)..n {
        adx[i] = ((adx[i - 1] * (window as f32 - 1.0)) + dx[i]) / window as f32;
    }

    adx
}

/// Computes the short-term vs long-term EMA ratio of volume:
///
/// ```text
/// EMA_short_t = α * Volume_t + (1 - α) * EMA_{t-1}
/// EMA_long_t  = α_long * Volume_t + (1 - α_long) * EMA_{t-1}
/// Vol_EMA_Ratio_t = EMA_short_t / EMA_long_t
/// ```
///
/// The first `long_window` timesteps are set to 0.0.
fn volume_ema_ratio(volumes: &[f32], short_window: usize, long_window: usize) -> Vec<f32> {
    let ema_short = ema_ratio(volumes, short_window);
    let ema_long = ema_ratio(volumes, long_window);
    ema_short
        .iter()
        .zip(&ema_long)
        .map(|(s, l)| s / l.max(EPS))
        .collect()
}

/// Computes volume spikes: ratio of current volume to rolling median:` Volume_Spike_t = Volume_t / median(Volume_{t-window..t-1})`
///
/// The first `window` timesteps are set to 0.0.
fn volume_spike(volumes: &[f32], window: usize) -> Vec<f32> {
    let n = volumes.len();
    let mut out = vec![0.0; n];

    for i in window..n {
        let slice = &volumes[i - window..i];
        let mut s = slice.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if window % 2 == 1 {
            // odd-length window: pick the middle
            s[window / 2]
        } else {
            // even-length window: average the two middle values
            (s[window / 2 - 1] + s[window / 2]) / 2.0
        };

        out[i] = volumes[i] / median.max(EPS);
    }

    out
}

/// Computes the On-Balance Volume (OBV), a cumulative volume trend indicator:
/// ```text
/// OBV_0 = 0
/// OBV_t = OBV_{t-1} +
///     +Volume_t if Close_t > Close_{t-1}
///     -Volume_t if Close_t < Close_{t-1}
///      0       if Close_t == Close_{t-1}
/// ```
///
/// The first timestep (t=0) is set to 0.0.
fn obv(closes: &[f32], volumes: &[f32]) -> Vec<f32> {
    let n = closes.len();
    let mut obv = vec![0.0; n];
    for i in 1..n {
        obv[i] = obv[i - 1]
            + if closes[i] > closes[i - 1] {
                volumes[i]
            } else if closes[i] < closes[i - 1] {
                -volumes[i]
            } else {
                0.0
            };
    }
    obv
}
