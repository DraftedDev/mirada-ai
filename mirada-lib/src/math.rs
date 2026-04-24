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

/// Processes the raw input data into features.
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

    let ema10 = ema_ratio(closes, 10);
    let ema20 = ema_ratio(closes, 20);

    let ema_diff = ema10
        .iter()
        .zip(&ema20)
        .map(|(e10, e20)| *e10 - *e20)
        .collect::<Vec<_>>();

    let momentum5 = closes
        .iter()
        .zip(closes.iter().skip(5))
        .map(|(c0, c5)| (c5 / c0) - 1.0)
        .collect::<Vec<_>>();

    let (body, upper, lower, range) = candle_features(opens, closes, highs, lows);

    let vol_ratio = volume_ratio(volumes, 10);
    let vol_spike = volume_spike(volumes, 5);
    let obv = obv(closes, volumes);

    let z_score_price_vs_sma = z_score_price_vs_sma(closes, 20);
    let volat_regime = volatility_regime(&lr1, 10, 20);
    let trend_strength_vs_noise = trend_strength_vs_noise(closes, highs, lows, 5, 10, 10);

    let rsi = rsi(closes, 7);
    let macd = macd_histogram(closes, 8, 17, 5);
    let atr14 = atr(highs, lows, closes, 14);

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
            ema10[i],
            ema20[i],
            ema_diff[i],
            momentum5[i],
            body[i],
            upper[i],
            lower[i],
            range[i],
            vol_ratio[i],
            vol_spike[i],
            obv[i],
            z_score_price_vs_sma[i],
            volat_regime[i],
            trend_strength_vs_noise[i],
            rsi[i],
            macd[i],
            atr14[i],
        ];
    }

    out
}

/// Normalize the output from [process] into rolling mean/std and clipped data.
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
