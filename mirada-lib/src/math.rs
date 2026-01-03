use crate::consts::feature_args::{
    ATR_PERIOD, BOLL_BAND_NUM_STD, BOLL_BAND_WINDOW, EMA_DIST_SPAN, MACD_HIST_LONG,
    MACD_HIST_SHORT, MACD_HIST_SIGNAL, MOM_PERIOD, PR_VS_LONG_SMA_WINDOW, ROLL_VOLAT_WINDOW,
    RSI_PERIOD, SMA_DIST_WINDOW, VOL_Z_SC_WINDOW, VOLAT_RAT_LONG, VOLAT_RAT_SHORT,
};
use crate::consts::feature_index::{
    ATR_IDX, ATR_RAT_IDX, BOLL_BAND_IDX, EMA_DIST_IDX, HL_RANGE_IDX, LOG_RET_IDX, MACD_HIST_IDX,
    MOM_IDX, OC_RET_IDX, PR_VOL_PRE_IDX, PR_VS_LONG_SMA_IDX, ROLL_VOLAT_IDX, RSI_IDX, SMA_DIST_IDX,
    TRUE_RANGES_IDX, VOL_CHANGE_IDX, VOL_U_SC_IDX, VOLAT_RAT_IDX,
};
use crate::consts::{CLIP, EPS, FEATURE_SIZE, WINDOW_SCALE, WINDOW_Z};

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
    const SKIPPED_TIMESTEPS: usize = 30;

    let n = closes.len();
    assert!(
        opens.len() == n && volumes.len() == n && highs.len() == n && lows.len() == n,
        "Input vectors must have the same length"
    );

    assert!(
        n > SKIPPED_TIMESTEPS,
        "Not enough timesteps to process. Need at least {SKIPPED_TIMESTEPS}, but got {n}."
    );

    // Compute all feature vectors (each helper returns a Vec<f32> length n)
    let log_returns = compute_log_returns(&closes);

    let volume_changes = compute_volume_change(&volumes);
    let volume_z_score_20 = compute_volume_z_score(&volumes, VOL_Z_SC_WINDOW);
    let price_volume_pressure = compute_price_volume_pressure(&log_returns, &volume_changes);

    let rolling_volatility_10 = compute_rolling_std(&log_returns, ROLL_VOLAT_WINDOW);
    let volatility_ratio = compute_volatility_ratio(&log_returns, VOLAT_RAT_SHORT, VOLAT_RAT_LONG);

    let high_low_ranges = compute_high_low_range(&highs, &lows, &closes);
    let open_close_returns = compute_open_close_return(&opens, &closes);

    let sma_dist_10 = compute_sma_distance(&closes, SMA_DIST_WINDOW);
    let ema_dist_20 = compute_ema_distance(&closes, EMA_DIST_SPAN);
    let price_vs_long_sma = compute_sma_distance(&closes, PR_VS_LONG_SMA_WINDOW);

    let momentum_10 = compute_momentum(&closes, MOM_PERIOD);
    let true_ranges = compute_true_range(&highs, &lows, &closes);

    let atr_14 = compute_atr(&true_ranges, ATR_PERIOD);
    let atr_ratio = compute_atr_ratio(&atr_14, &closes);

    let bollinger_20 = compute_bollinger_distance(&closes, BOLL_BAND_WINDOW, BOLL_BAND_NUM_STD);
    let rsi_14 = compute_rsi(&closes, RSI_PERIOD);
    let macd_histogram =
        compute_macd_histogram(&closes, MACD_HIST_SHORT, MACD_HIST_LONG, MACD_HIST_SIGNAL);

    // Assemble final output
    let mut out = vec![[0.0_f32; FEATURE_SIZE]; n];

    for i in 0..n {
        out[i][LOG_RET_IDX] = log_returns[i];
        out[i][VOL_CHANGE_IDX] = volume_changes[i];
        out[i][VOL_U_SC_IDX] = volume_z_score_20[i];
        out[i][PR_VS_LONG_SMA_IDX] = price_volume_pressure[i];
        out[i][ROLL_VOLAT_IDX] = rolling_volatility_10[i];
        out[i][VOLAT_RAT_IDX] = volatility_ratio[i];
        out[i][HL_RANGE_IDX] = high_low_ranges[i];
        out[i][OC_RET_IDX] = open_close_returns[i];
        out[i][SMA_DIST_IDX] = sma_dist_10[i];
        out[i][EMA_DIST_IDX] = ema_dist_20[i];
        out[i][PR_VS_LONG_SMA_IDX] = price_vs_long_sma[i];
        out[i][MOM_IDX] = momentum_10[i];
        out[i][TRUE_RANGES_IDX] = true_ranges[i];
        out[i][ATR_IDX] = atr_14[i];
        out[i][ATR_RAT_IDX] = atr_ratio[i];
        out[i][BOLL_BAND_IDX] = bollinger_20[i];
        out[i][RSI_IDX] = rsi_14[i];
        out[i][MACD_HIST_IDX] = macd_histogram[i];
    }

    out
}

/// Normalize the output from [process].
///
/// - log_return, volume_change, high_low_range, open_close_return,
///   SMA distance, EMA distance, price vs long SMA, momentum,
///   price x volume pressure, volume z-score -> rolling z-score
/// - rolling_volatility, ATR, ATR ratio -> divide by long-term mean
/// - Bollinger distance, RSI, MACD histogram -> leave as-is
pub fn normalize(features: &Vec<[f32; FEATURE_SIZE]>) -> Vec<[f32; FEATURE_SIZE]> {
    let n = features.len();
    let mut out = vec![[0.0; FEATURE_SIZE]; n];

    // Extract per-feature vectors
    let mut log_return: Vec<f32> = features.iter().map(|x| x[LOG_RET_IDX]).collect();
    let mut volume_change: Vec<f32> = features.iter().map(|x| x[VOL_CHANGE_IDX]).collect();
    let mut volume_z_score: Vec<f32> = features.iter().map(|x| x[VOL_U_SC_IDX]).collect();
    let mut price_volume_pressure: Vec<f32> = features.iter().map(|x| x[PR_VOL_PRE_IDX]).collect();
    let mut rolling_vol: Vec<f32> = features.iter().map(|x| x[ROLL_VOLAT_IDX]).collect();
    let volatility_ratio: Vec<f32> = features.iter().map(|x| x[VOLAT_RAT_IDX]).collect();
    let mut hl_range: Vec<f32> = features.iter().map(|x| x[HL_RANGE_IDX]).collect();
    let mut oc_return: Vec<f32> = features.iter().map(|x| x[OC_RET_IDX]).collect();
    let mut sma_dist: Vec<f32> = features.iter().map(|x| x[SMA_DIST_IDX]).collect();
    let mut ema_dist: Vec<f32> = features.iter().map(|x| x[EMA_DIST_IDX]).collect();
    let mut price_vs_long_sma: Vec<f32> = features.iter().map(|x| x[PR_VS_LONG_SMA_IDX]).collect();
    let mut momentum: Vec<f32> = features.iter().map(|x| x[MOM_IDX]).collect();
    let true_ranges: Vec<f32> = features.iter().map(|x| x[TRUE_RANGES_IDX]).collect();
    let mut atr: Vec<f32> = features.iter().map(|x| x[ATR_IDX]).collect();
    let mut atr_ratio: Vec<f32> = features.iter().map(|x| x[ATR_RAT_IDX]).collect();
    let bollinger: Vec<f32> = features.iter().map(|x| x[BOLL_BAND_IDX]).collect(); // leave as-is
    let rsi: Vec<f32> = features.iter().map(|x| x[RSI_IDX]).collect(); // leave as-is
    let macd_hist: Vec<f32> = features.iter().map(|x| x[MACD_HIST_IDX]).collect(); // leave as-is

    // Normalize rolling z-score features
    log_return = rolling_z_score(&log_return, WINDOW_Z, CLIP);
    volume_change = rolling_z_score(&volume_change, WINDOW_Z, CLIP);
    volume_z_score = rolling_z_score(&volume_z_score, WINDOW_Z, CLIP);
    price_volume_pressure = rolling_z_score(&price_volume_pressure, WINDOW_Z, CLIP);
    hl_range = rolling_z_score(&hl_range, WINDOW_Z, CLIP);
    oc_return = rolling_z_score(&oc_return, WINDOW_Z, CLIP);
    sma_dist = rolling_z_score(&sma_dist, WINDOW_Z, CLIP);
    ema_dist = rolling_z_score(&ema_dist, WINDOW_Z, CLIP);
    price_vs_long_sma = rolling_z_score(&price_vs_long_sma, WINDOW_Z, CLIP);
    momentum = rolling_z_score(&momentum, WINDOW_Z, CLIP);

    // Normalize volatility / ATR
    rolling_vol = divide_by_long_mean(&rolling_vol, WINDOW_SCALE);
    atr = divide_by_long_mean(&atr, WINDOW_SCALE);
    atr_ratio = divide_by_long_mean(&atr_ratio, WINDOW_SCALE);

    // Combine back into output
    for i in 0..n {
        out[i][LOG_RET_IDX] = log_return[i];
        out[i][VOL_CHANGE_IDX] = volume_change[i];
        out[i][VOL_U_SC_IDX] = volume_z_score[i];
        out[i][PR_VS_LONG_SMA_IDX] = price_volume_pressure[i];
        out[i][ROLL_VOLAT_IDX] = rolling_vol[i];
        out[i][VOLAT_RAT_IDX] = volatility_ratio[i];
        out[i][HL_RANGE_IDX] = hl_range[i];
        out[i][OC_RET_IDX] = oc_return[i];
        out[i][SMA_DIST_IDX] = sma_dist[i];
        out[i][EMA_DIST_IDX] = ema_dist[i];
        out[i][PR_VS_LONG_SMA_IDX] = price_vs_long_sma[i];
        out[i][MOM_IDX] = momentum[i];
        out[i][TRUE_RANGES_IDX] = true_ranges[i];
        out[i][ATR_IDX] = atr[i];
        out[i][ATR_RAT_IDX] = atr_ratio[i];
        out[i][BOLL_BAND_IDX] = bollinger[i];
        out[i][RSI_IDX] = rsi[i];
        out[i][MACD_HIST_IDX] = macd_hist[i];
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
        let std = var.sqrt().max(EPS);
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
        out[i] = values[i] / mean.max(EPS);
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
