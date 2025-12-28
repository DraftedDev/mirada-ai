use crate::data::FEATURE_SIZE;

/// Small epsilon to avoid division by zero.
const EPS: f32 = 1e-8;

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

/// Normalize the output from [process].
///
/// - log_return, volume_change, high_low_range, open_close_return,
///   SMA distance, EMA distance, price vs long SMA, momentum,
///   price x volume pressure, volume z-score -> rolling z-score
/// - rolling_volatility, ATR, ATR ratio -> divide by long-term mean
/// - Bollinger distance, RSI, MACD histogram -> leave as-is
///
/// # Arguments
/// `window_z` - rolling window for z-score normalization (e.g. 60)
/// `window_scale` - long-term mean window for volatility/ATR (e.g. 252)
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
    let mut volume_z_score: Vec<f32> = features.iter().map(|x| x[2]).collect();
    let mut price_volume_pressure: Vec<f32> = features.iter().map(|x| x[3]).collect();
    let mut rolling_vol: Vec<f32> = features.iter().map(|x| x[4]).collect();
    let volatility_ratio: Vec<f32> = features.iter().map(|x| x[5]).collect();
    let mut hl_range: Vec<f32> = features.iter().map(|x| x[6]).collect();
    let mut oc_return: Vec<f32> = features.iter().map(|x| x[7]).collect();
    let mut sma_dist: Vec<f32> = features.iter().map(|x| x[8]).collect();
    let mut ema_dist: Vec<f32> = features.iter().map(|x| x[9]).collect();
    let mut price_vs_long_sma: Vec<f32> = features.iter().map(|x| x[10]).collect();
    let mut momentum: Vec<f32> = features.iter().map(|x| x[11]).collect();
    let true_ranges: Vec<f32> = features.iter().map(|x| x[12]).collect();
    let mut atr: Vec<f32> = features.iter().map(|x| x[13]).collect();
    let mut atr_ratio: Vec<f32> = features.iter().map(|x| x[14]).collect();
    let bollinger: Vec<f32> = features.iter().map(|x| x[15]).collect(); // leave as-is
    let rsi: Vec<f32> = features.iter().map(|x| x[16]).collect(); // leave as-is
    let macd_hist: Vec<f32> = features.iter().map(|x| x[17]).collect(); // leave as-is

    // Normalize rolling z-score features
    log_return = rolling_z_score(&log_return, window_z, clip);
    volume_change = rolling_z_score(&volume_change, window_z, clip);
    volume_z_score = rolling_z_score(&volume_z_score, window_z, clip);
    price_volume_pressure = rolling_z_score(&price_volume_pressure, window_z, clip);
    hl_range = rolling_z_score(&hl_range, window_z, clip);
    oc_return = rolling_z_score(&oc_return, window_z, clip);
    sma_dist = rolling_z_score(&sma_dist, window_z, clip);
    ema_dist = rolling_z_score(&ema_dist, window_z, clip);
    price_vs_long_sma = rolling_z_score(&price_vs_long_sma, window_z, clip);
    momentum = rolling_z_score(&momentum, window_z, clip);

    // Normalize volatility / ATR
    rolling_vol = divide_by_long_mean(&rolling_vol, window_scale);
    atr = divide_by_long_mean(&atr, window_scale);
    atr_ratio = divide_by_long_mean(&atr_ratio, window_scale);

    // Combine back into output
    for i in 0..n {
        out[i][0] = log_return[i];
        out[i][1] = volume_change[i];
        out[i][2] = volume_z_score[i];
        out[i][3] = price_volume_pressure[i];
        out[i][4] = rolling_vol[i];
        out[i][5] = volatility_ratio[i];
        out[i][6] = hl_range[i];
        out[i][7] = oc_return[i];
        out[i][8] = sma_dist[i];
        out[i][9] = ema_dist[i];
        out[i][10] = price_vs_long_sma[i];
        out[i][11] = momentum[i];
        out[i][12] = true_ranges[i];
        out[i][13] = atr[i];
        out[i][14] = atr_ratio[i];
        out[i][15] = bollinger[i];
        out[i][16] = rsi[i];
        out[i][17] = macd_hist[i];
    }

    out
}

/// Processes the input data into:
/// - log returns
/// - volume changes
/// - volume z score (20)
/// - price volume pressure
/// - rolling volatility (10)
/// - volatility ratio (10, 50)
/// - high low ranges
/// - open close returns
/// - sma distance (10)
/// - ema distance (20)
/// - price vs long sma (50)
/// - momentum (10)
/// - true ranges
/// - atr (14)
/// - atr ratio (14)
/// - bollinger band distance (20)
/// - rsi (14)
/// - macd histogram (12, 16, 9)
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

    // Compute all feature vectors (each helper returns a Vec<f32> length n)
    let log_returns = compute_log_returns(&closes);

    let volume_changes = compute_volume_change(&volumes);
    let volume_z_score_20 = compute_volume_z_score(&volumes, 20);
    let price_volume_pressure = compute_price_volume_pressure(&log_returns, &volume_changes);

    let rolling_volatility_10 = compute_rolling_std(&log_returns, 10);
    let volatility_ratio = compute_volatility_ratio(&log_returns, 10, 50);

    let high_low_ranges = compute_high_low_range(&highs, &lows, &closes);
    let open_close_returns = compute_open_close_return(&opens, &closes);

    let sma_dist_10 = compute_sma_distance(&closes, 10);
    let ema_dist_20 = compute_ema_distance(&closes, 20);
    let price_vs_long_sma = compute_sma_distance(&closes, 50);

    let momentum_10 = compute_momentum(&closes, 10);
    let true_ranges = compute_true_range(&highs, &lows, &closes);

    let atr_14 = compute_atr(&true_ranges, 14);
    let atr_ratio = compute_atr_ratio(&atr_14, &closes);

    let bollinger_20 = compute_bollinger_distance(&closes, 20, 2.0);
    let rsi_14 = compute_rsi(&closes, 14);
    let macd_histogram = compute_macd_histogram(&closes, 12, 16, 9);

    // Assemble final output
    let mut out = vec![[0.0_f32; FEATURE_SIZE]; n];

    for i in 0..n {
        out[i][0] = log_returns[i];
        out[i][1] = volume_changes[i];
        out[i][2] = volume_z_score_20[i];
        out[i][3] = price_volume_pressure[i];
        out[i][4] = rolling_volatility_10[i];
        out[i][5] = volatility_ratio[i];
        out[i][6] = high_low_ranges[i];
        out[i][7] = open_close_returns[i];
        out[i][8] = sma_dist_10[i];
        out[i][9] = ema_dist_20[i];
        out[i][10] = price_vs_long_sma[i];
        out[i][11] = momentum_10[i];
        out[i][12] = true_ranges[i];
        out[i][13] = atr_14[i];
        out[i][14] = atr_ratio[i];
        out[i][15] = bollinger_20[i];
        out[i][16] = rsi_14[i];
        out[i][17] = macd_histogram[i];
    }

    out
}

/// Helper for rolling z-score with clipping.
fn rolling_z_score(values: &[f32], window: usize, clip: f32) -> Vec<f32> {
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

/// Helper for dividing by long-term mean.
fn divide_by_long_mean(values: &[f32], window: usize) -> Vec<f32> {
    let n = values.len();
    let mut out = vec![0.0; n];
    for i in window..n {
        let mean = values[i - window..i].iter().sum::<f32>() / window as f32;
        out[i] = values[i] / mean.max(1e-6);
    }
    out
}

/// Compute log returns: ln(close_t / close_{t-1}).
/// Returns a Vec of length n; index 0 is 0.0.
fn compute_log_returns(closes: &[f32]) -> Vec<f32> {
    let n = closes.len();
    let mut out = vec![0.0_f32; n];
    for i in 1..n {
        let prev = closes[i - 1].max(EPS);
        out[i] = (closes[i] / prev).ln();
    }
    out
}

/// Compute volume change: (vol_t - vol_{t-1}) / vol_{t-1}
/// Returns length n; index 0 is 0.0.
fn compute_volume_change(volumes: &[f32]) -> Vec<f32> {
    let n = volumes.len();
    let mut out = vec![0.0_f32; n];
    for i in 1..n {
        let prev = volumes[i - 1].max(EPS);
        out[i] = (volumes[i] - volumes[i - 1]) / prev;
    }
    out
}

/// High-low range normalized by close: (high - low) / close
/// Returns length n; index 0 may be set to 0.0.
fn compute_high_low_range(highs: &[f32], lows: &[f32], closes: &[f32]) -> Vec<f32> {
    let n = closes.len();
    let mut out = vec![0.0_f32; n];
    for i in 0..n {
        let c = closes[i].max(EPS);
        out[i] = (highs[i] - lows[i]) / c;
    }
    out
}

/// Open-Close return: (close - open) / open
fn compute_open_close_return(opens: &[f32], closes: &[f32]) -> Vec<f32> {
    let n = closes.len();
    let mut out = vec![0.0_f32; n];
    for i in 0..n {
        let o = opens[i].max(EPS);
        out[i] = (closes[i] - opens[i]) / o;
    }
    out
}

/// Rolling standard deviation (population std) of `values` over `window` periods.
/// For indices < window - 1 the result will be 0.0.
fn compute_rolling_std(values: &[f32], window: usize) -> Vec<f32> {
    let n = values.len();
    let mut out = vec![0.0_f32; n];
    if window == 0 || n == 0 {
        return out;
    }
    for i in (window - 1)..n {
        let slice = &values[i + 1 - window..=i];
        let mean = slice.iter().sum::<f32>() / window as f32;
        let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / window as f32;
        out[i] = var.sqrt();
    }
    out
}

/// Simple SMA distance: (close - sma) / sma, where sma is simple moving average over `window`.
/// For indices < window - 1 returns 0.0.
fn compute_sma_distance(closes: &[f32], window: usize) -> Vec<f32> {
    let n = closes.len();
    let mut out = vec![0.0_f32; n];
    if window == 0 || n == 0 {
        return out;
    }
    for i in (window - 1)..n {
        let slice = &closes[i + 1 - window..=i];
        let sma = slice.iter().sum::<f32>() / window as f32;
        out[i] = (closes[i] - sma) / sma.max(EPS);
    }
    out
}

/// EMA distance: compute EMA with given span and return (close - ema) / ema.
/// Uses standard smoothing alpha = 2/(span + 1).
/// For early indices (i < span-1) returns 0.0.
fn compute_ema_distance(closes: &[f32], span: usize) -> Vec<f32> {
    let n = closes.len();
    let mut out = vec![0.0_f32; n];
    if span == 0 || n == 0 {
        return out;
    }
    let alpha = 2.0 / (span as f32 + 1.0);
    let mut ema = closes[0];
    for i in 1..n {
        ema = alpha * closes[i] + (1.0 - alpha) * ema;
        if i + 1 >= span {
            out[i] = (closes[i] - ema) / ema.max(EPS);
        }
    }
    out
}

/// Momentum: (close_t / close_{t-period}) - 1.0
/// For i < period returns 0.0.
fn compute_momentum(closes: &[f32], period: usize) -> Vec<f32> {
    let n = closes.len();
    let mut out = vec![0.0_f32; n];
    if period == 0 || n == 0 {
        return out;
    }
    for i in period..n {
        out[i] = closes[i] / closes[i - period].max(EPS) - 1.0;
    }
    out
}

/// True Range per-timestep: max(high-low, |high - prev_close|, |low - prev_close|)
/// Index 0 is set to high[0] - low[0]
fn compute_true_range(highs: &[f32], lows: &[f32], closes: &[f32]) -> Vec<f32> {
    let n = closes.len();
    let mut tr = vec![0.0_f32; n];
    if n == 0 {
        return tr;
    }
    tr[0] = (highs[0] - lows[0]).max(EPS);
    for i in 1..n {
        let a = highs[i] - lows[i];
        let b = (highs[i] - closes[i - 1]).abs();
        let c = (lows[i] - closes[i - 1]).abs();
        tr[i] = a.max(b).max(c);
    }
    tr
}

/// ATR: simple moving average of true range over `period`.
/// For i < period - 1 returns 0.0.
fn compute_atr(true_range: &[f32], period: usize) -> Vec<f32> {
    let n = true_range.len();
    let mut out = vec![0.0_f32; n];
    if period == 0 || n == 0 {
        return out;
    }
    for i in (period - 1)..n {
        let slice = &true_range[i + 1 - period..=i];
        out[i] = slice.iter().sum::<f32>() / period as f32;
    }
    out
}

/// Bollinger band distance: (close - mean) / (num_std * std)
/// Returns 0.0 for i < window - 1 or if bandwidth is zero.
fn compute_bollinger_distance(closes: &[f32], window: usize, num_std: f32) -> Vec<f32> {
    let n = closes.len();
    let mut out = vec![0.0_f32; n];
    if window == 0 || n == 0 {
        return out;
    }
    for i in (window - 1)..n {
        let slice = &closes[i + 1 - window..=i];
        let mean = slice.iter().sum::<f32>() / window as f32;
        let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / window as f32;
        let std = var.sqrt();
        let band_width = num_std * std;
        if band_width > EPS {
            out[i] = (closes[i] - mean) / band_width;
        } else {
            out[i] = 0.0;
        }
    }
    out
}

/// Relative Strength Index (RSI).
/// Uses simple moving average of gains and losses over `period`.
/// Returns normalized RSI: (rsi - 50) / 50.
/// For i < period returns 0.0.
fn compute_rsi(closes: &[f32], period: usize) -> Vec<f32> {
    let n = closes.len();
    let mut out = vec![0.0; n];
    if period == 0 || n < period + 1 {
        return out;
    }

    let mut gains = vec![0.0; n];
    let mut losses = vec![0.0; n];

    for i in 1..n {
        let delta = closes[i] - closes[i - 1];
        if delta > 0.0 {
            gains[i] = delta;
        } else {
            losses[i] = -delta;
        }
    }

    for i in period..n {
        let avg_gain = gains[i + 1 - period..=i].iter().sum::<f32>() / period as f32;
        let avg_loss = losses[i + 1 - period..=i].iter().sum::<f32>() / period as f32;

        if avg_loss > EPS {
            let rs = avg_gain / avg_loss;
            let rsi = 100.0 - (100.0 / (1.0 + rs));
            out[i] = (rsi - 50.0) / 50.0;
        }
    }

    out
}

/// MACD histogram: EMA(short) - EMA(long) minus signal EMA.
/// Returns MACD histogram; early indices return 0.0.
fn compute_macd_histogram(closes: &[f32], short: usize, long: usize, signal: usize) -> Vec<f32> {
    let n = closes.len();
    let mut out = vec![0.0; n];
    if short == 0 || long == 0 || signal == 0 || n == 0 {
        return out;
    }

    let alpha_s = 2.0 / (short as f32 + 1.0);
    let alpha_l = 2.0 / (long as f32 + 1.0);
    let alpha_sig = 2.0 / (signal as f32 + 1.0);

    let mut ema_s = closes[0];
    let mut ema_l = closes[0];
    let mut sig = 0.0;

    for i in 1..n {
        ema_s = alpha_s * closes[i] + (1.0 - alpha_s) * ema_s;
        ema_l = alpha_l * closes[i] + (1.0 - alpha_l) * ema_l;

        let macd = ema_s - ema_l;

        sig = alpha_sig * macd + (1.0 - alpha_sig) * sig;

        if i + 1 >= long + signal {
            out[i] = macd - sig;
        }
    }

    out
}

/// Volatility ratio: short_window_std / long_window_std.
///
/// Returns 0.0 where long volatility is zero or insufficient data.
fn compute_volatility_ratio(
    log_returns: &[f32],
    short_window: usize,
    long_window: usize,
) -> Vec<f32> {
    let n = log_returns.len();
    let mut out = vec![0.0; n];

    if short_window == 0 || long_window == 0 {
        return out;
    }

    let short_vol = compute_rolling_std(log_returns, short_window);
    let long_vol = compute_rolling_std(log_returns, long_window);

    for i in 0..n {
        if long_vol[i] > EPS {
            out[i] = short_vol[i] / long_vol[i];
        }
    }

    out
}

/// ATR ratio: atr / close.
fn compute_atr_ratio(atr: &[f32], closes: &[f32]) -> Vec<f32> {
    let n = closes.len();
    let mut out = vec![0.0; n];
    for i in 0..n {
        out[i] = atr[i] / closes[i].max(EPS);
    }
    out
}

/// Volume z-score over `window`.
/// For i < window - 1 returns 0.0.
fn compute_volume_z_score(volumes: &[f32], window: usize) -> Vec<f32> {
    let n = volumes.len();
    let mut out = vec![0.0; n];
    if window == 0 || n == 0 {
        return out;
    }

    for i in (window - 1)..n {
        let slice = &volumes[i + 1 - window..=i];
        let mean = slice.iter().sum::<f32>() / window as f32;
        let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / window as f32;
        let std = var.sqrt();

        if std > EPS {
            out[i] = (volumes[i] - mean) / std;
        }
    }

    out
}

/// Price × Volume pressure: log_return * volume_change
fn compute_price_volume_pressure(log_returns: &[f32], volume_change: &[f32]) -> Vec<f32> {
    let n = log_returns.len();
    let mut out = vec![0.0; n];
    for i in 0..n {
        out[i] = log_returns[i] * volume_change[i];
    }
    out
}
