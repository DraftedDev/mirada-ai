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
// TODO: make this configurable via constants or whatever
pub fn normalize(features: &[[f32; FEATURE_SIZE]]) -> Vec<[f32; FEATURE_SIZE]> {
    let n = features.len();
    let mut out = vec![[0.0; FEATURE_SIZE]; n];
    let window = ROLLING_WINDOW;

    for t in 0..n {
        if t < window {
            continue;
        }
        let slice = &features[t - window..t];
        let mean: [f32; FEATURE_SIZE] = slice
            .iter()
            .fold([0.0; FEATURE_SIZE], |mut acc, row| {
                for f in 0..FEATURE_SIZE {
                    acc[f] += row[f];
                }
                acc
            })
            .map(|v| v / window as f32);

        let std: [f32; FEATURE_SIZE] = slice
            .iter()
            .fold([0.0; FEATURE_SIZE], |mut acc, row| {
                for f in 0..FEATURE_SIZE {
                    acc[f] += (row[f] - mean[f]).powi(2);
                }
                acc
            })
            .map(|v| (v / window as f32).sqrt().max(EPS));

        for f in 0..FEATURE_SIZE {
            out[t][f] = ((features[t][f] - mean[f]) / std[f]).clamp(-CLIP, CLIP);
        }
    }

    out
}

/// Processes the input data into features.
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
    let lr2 = log_return(closes, 2);
    let lr3 = log_return(closes, 3);

    let vol2 = rolling_std(&lr1, 2);
    let vol5 = rolling_std(&lr1, 5);
    let vol10 = rolling_std(&lr1, 10);

    let sma3 = sma_ratio(closes, 3);
    let sma5 = sma_ratio(closes, 5);
    let ema5 = ema_ratio(closes, 5);
    let ema12 = ema_ratio(closes, 12);

    let (body, upper, lower, range) = candle_features(opens, closes, highs, lows);

    let vol_ratio = volume_ratio(volumes, 14);
    let vol_change = volume_change(volumes);
    let vol_ema_rat = volume_ema_ratio(volumes, 8, 12);
    let vol_spike = volume_spike(volumes, 8);

    let atr7 = atr(highs, lows, closes, 7);
    let adx3 = adx(highs, lows, closes, 3);
    let obv = obv(closes, volumes);
    let z_score_price_vs_sma = z_score_price_vs_sma(closes, 12);
    let bollinger_band_pos = bollinger_position(closes, 12);
    let volat_regime = volatility_regime(&lr1, 10, 20);
    let trend_strength_vs_noise = trend_strength_vs_noise(closes, highs, lows, 8, 12, 10);
    let rsi5 = rsi(closes, 5);
    let macd_hist = macd_histogram(closes, 5, 12, 3);

    let mut out = vec![[0.0; FEATURE_SIZE]; n];

    for i in 0..n {
        out[i] = [
            lr1[i],
            lr2[i],
            lr3[i],
            vol2[i],
            vol5[i],
            vol10[i],
            sma3[i],
            sma5[i],
            ema5[i],
            ema12[i],
            body[i],
            upper[i],
            lower[i],
            range[i],
            vol_ratio[i],
            vol_change[i],
            vol_ema_rat[i],
            vol_spike[i],
            atr7[i],
            adx3[i],
            obv[i],
            z_score_price_vs_sma[i],
            bollinger_band_pos[i],
            volat_regime[i],
            trend_strength_vs_noise[i],
            rsi5[i],
            macd_hist[i],
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

/// Computes the Z-score of the current price relative to a rolling SMA:
/// `z_t = (Close_t - mean(Close_{t-window..t-1})) / std(Close_{t-window..t-1})`
///
/// The first `window` timesteps are set to 0.0.
fn z_score_price_vs_sma(closes: &[f32], window: usize) -> Vec<f32> {
    let n = closes.len();
    let mut out = vec![0.0; n];

    for i in window..n {
        let slice = &closes[i - window..i];

        let mean = slice.iter().sum::<f32>() / window as f32;
        let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / window as f32;
        let std = var.sqrt().max(EPS);

        out[i] = (closes[i] - mean) / std;
    }

    out
}

/// Computes normalized position of price inside Bollinger Bands: `BB_Pos_t = (Close_t - SMA_t) / (2 * std_t)`
///
/// The first `window` timesteps are set to 0.0.
fn bollinger_position(closes: &[f32], window: usize) -> Vec<f32> {
    let n = closes.len();
    let mut out = vec![0.0; n];

    for i in window..n {
        let slice = &closes[i - window..i];

        let mean = slice.iter().sum::<f32>() / window as f32;
        let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / window as f32;
        let std = var.sqrt().max(EPS);

        out[i] = (closes[i] - mean) / (2.0 * std);
    }

    out
}

/// Measures short-term volatility relative to long-term volatility: `regime_t = std_short(returns) / std_long(returns)`
///
/// The first `long` timesteps are set to 0.0.
fn volatility_regime(returns: &[f32], short: usize, long: usize) -> Vec<f32> {
    let short_std = rolling_std(returns, short);
    let long_std = rolling_std(returns, long);

    short_std
        .iter()
        .zip(long_std.iter())
        .map(|(s, l)| s / l.max(EPS))
        .collect()
}

/// Measures directional trend strength relative to market noise (ATR): `TrendNoise_t = |SMA_short - SMA_long| / ATR`
///
/// The first `long` timesteps are set to 0.0.
fn trend_strength_vs_noise(
    closes: &[f32],
    highs: &[f32],
    lows: &[f32],
    short: usize,
    long: usize,
    atr_window: usize,
) -> Vec<f32> {
    let n = closes.len();
    let mut out = vec![0.0; n];

    let sma_short = rolling_mean(closes, short);
    let sma_long = rolling_mean(closes, long);
    let atr_vals = atr(highs, lows, closes, atr_window);

    for i in 0..n {
        out[i] = (sma_short[i] - sma_long[i]).abs() / atr_vals[i].max(EPS);
    }

    out
}

/// Computes the Relative Strength Index using Wilder's smoothing:
/// ```text
/// RS  = AvgGain / AvgLoss
/// RSI = 100 - (100 / (1 + RS))
/// ```
///
/// The first `window` timesteps are set to 0.0.
fn rsi(closes: &[f32], window: usize) -> Vec<f32> {
    let n = closes.len();
    let mut out = vec![0.0; n];

    if n <= window {
        return out;
    }

    let mut gains = vec![0.0; n];
    let mut losses = vec![0.0; n];

    for i in 1..n {
        let diff = closes[i] - closes[i - 1];
        if diff > 0.0 {
            gains[i] = diff;
        } else {
            losses[i] = -diff;
        }
    }

    // Initial SMA
    let mut avg_gain = gains[1..=window].iter().sum::<f32>() / window as f32;
    let mut avg_loss = losses[1..=window].iter().sum::<f32>() / window as f32;

    let compute_rsi = |g: f32, l: f32| {
        if l == 0.0 {
            100.0
        } else {
            let rs = g / l;
            100.0 - 100.0 / (1.0 + rs)
        }
    };

    out[window] = compute_rsi(avg_gain, avg_loss);

    for i in (window + 1)..n {
        avg_gain = (avg_gain * (window as f32 - 1.0) + gains[i]) / window as f32;
        avg_loss = (avg_loss * (window as f32 - 1.0) + losses[i]) / window as f32;

        out[i] = compute_rsi(avg_gain, avg_loss);
    }

    // ML scaling (better distribution)
    out.iter()
        .map(|v| ((v - 50.0) / 25.0).clamp(-2.0, 2.0))
        .collect()
}

/// Computes the MACD histogram for a series of closing prices.
///
/// The first `long_window` entries are set to 0.0
pub fn macd_histogram(
    closes: &[f32],
    short_window: usize,
    long_window: usize,
    signal_window: usize,
) -> Vec<f32> {
    let n = closes.len();
    let mut hist = vec![0.0; n];

    if n == 0 || long_window == 0 || short_window == 0 || signal_window == 0 {
        return hist;
    }

    let ema_short = ema_ratio(closes, short_window);
    let ema_long = ema_ratio(closes, long_window);

    let macd_line: Vec<f32> = ema_short
        .iter()
        .zip(&ema_long)
        .map(|(s, l)| s - l)
        .collect();

    let mut signal_line = vec![0.0; n];
    if n > signal_window {
        let alpha = 2.0 / (signal_window as f32 + 1.0);
        signal_line[signal_window - 1] = macd_line[signal_window - 1];
        for i in signal_window..n {
            signal_line[i] = alpha * macd_line[i] + (1.0 - alpha) * signal_line[i - 1];
        }
    }

    for i in 0..n {
        hist[i] = macd_line[i] - signal_line[i];
    }

    hist
}
