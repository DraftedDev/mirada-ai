use crate::consts::{CLIP, EPS, OTHER_STOCKS, ROLLING_WINDOW, THRESHOLD};

pub fn generate_targets(closes: &[f32], horizon: usize) -> Vec<i32> {
    let n = closes.len();
    assert!(horizon > 0 && n > horizon, "Invalid horizon");

    let mut targets = Vec::with_capacity(n - horizon);

    for t in 0..(n - horizon) {
        let c0 = closes[t];
        let c1 = closes[t + horizon];

        let ret = (c1 / c0).ln();

        let label = if ret > THRESHOLD {
            2
        } else if ret < -THRESHOLD {
            0
        } else {
            1
        };

        targets.push(label);
    }

    targets
}

/// Processes the raw input data into features.
pub fn process(
    opens: Vec<f32>,
    closes: Vec<f32>,
    closes_other: [Vec<f32>; OTHER_STOCKS],
    volumes: Vec<f32>,
    highs: Vec<f32>,
    lows: Vec<f32>,
) -> Vec<Vec<f32>> {
    let n = opens.len();

    let lr1 = log_return(&closes, 1);
    let lr5 = log_return(&closes, 5);

    let vol20 = rolling_std(&lr1, 20);

    let ema_trend = ema_diff(&closes);

    let volume_z = z_score(&volumes, 20);
    let volume_spike = volume_z.iter().map(|v| v.tanh()).collect::<Vec<_>>();

    let price_z = price_z_score(&closes);

    let (body, upper, lower) = candle_features(&opens, &closes, &highs, &lows);

    let range_norm = normalized_range(&highs, &lows, &closes);

    let mut out = vec![Vec::with_capacity(10 + OTHER_STOCKS * 2); n];

    for i in 0..n {
        out[i].extend_from_slice(&[
            lr1[i],
            lr5[i],
            vol20[i],
            ema_trend[i],
            upper[i],
            body[i],
            lower[i],
            range_norm[i],
            volume_z[i],
            volume_spike[i],
            price_z[i],
        ]);
    }

    for k in 0..OTHER_STOCKS {
        let lr1_o = log_return(&closes_other[k], 1);
        let trend_o = ema_diff(&closes_other[k]);

        for i in 0..n {
            out[i].push(lr1_o[i] - lr1[i]);
            out[i].push(trend_o[i] - ema_trend[i]);
        }
    }

    out
}

/// Normalize the output from [process] into rolling mean/std and clipped data.
pub fn normalize(features: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let n = features.len();
    let d = features[0].len();

    let mut out = vec![vec![0.0; d]; n];

    for j in 0..d {
        let mut mean_buf = vec![0.0; n];
        let mut std_buf = vec![1.0; n];

        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for i in 0..n {
            let x = features[i][j];

            sum += x;
            sum_sq += x * x;

            if i >= ROLLING_WINDOW {
                let old = features[i - ROLLING_WINDOW][j];
                sum -= old;
                sum_sq -= old * old;
            }

            if i >= ROLLING_WINDOW - 1 {
                let mean = sum / ROLLING_WINDOW as f32;
                let var = (sum_sq / ROLLING_WINDOW as f32) - mean * mean;

                mean_buf[i] = mean;
                std_buf[i] = var.sqrt().max(EPS);
            } else {
                mean_buf[i] = sum / (i as f32 + 1.0);
                std_buf[i] = 1.0;
            }
        }

        for i in 0..n {
            let z = (features[i][j] - mean_buf[i]) / std_buf[i];
            out[i][j] = z.clamp(-CLIP, CLIP);
        }
    }

    out
}

fn safe_div(a: f32, b: f32) -> f32 {
    a / (b.abs() + EPS)
}

fn log_return(x: &[f32], lag: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for i in lag..x.len() {
        out[i] = (x[i] / x[i - lag]).ln();
    }
    out
}

fn rolling_mean(x: &[f32], window: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    let mut sum = 0.0;

    for i in 0..x.len() {
        sum += x[i];
        if i >= window {
            sum -= x[i - window];
        }
        if i >= window - 1 {
            out[i] = sum / window as f32;
        }
    }
    out
}

fn rolling_std(x: &[f32], window: usize) -> Vec<f32> {
    let mean = rolling_mean(x, window);
    let mut out = vec![0.0; x.len()];

    for i in window..x.len() {
        let mut var = 0.0;
        for j in (i - window)..i {
            let d = x[j] - mean[i];
            var += d * d;
        }
        out[i] = (var / window as f32).sqrt();
    }
    out
}

fn z_score(x: &[f32], window: usize) -> Vec<f32> {
    let mean = rolling_mean(x, window);
    let std = rolling_std(x, window);

    x.iter()
        .enumerate()
        .map(|(i, &v)| safe_div(v - mean[i], std[i]))
        .collect()
}

fn ema(x: &[f32], period: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    let alpha = 2.0 / (period as f32 + 1.0);

    out[0] = x[0];

    for i in 1..x.len() {
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1];
    }

    out
}

fn ema_diff(closes: &[f32]) -> Vec<f32> {
    let ema10 = ema(closes, 10);
    let ema20 = ema(closes, 20);

    ema10
        .iter()
        .zip(&ema20)
        .zip(closes)
        .map(|((a, b), c)| safe_div(a - b, *c))
        .collect()
}

fn candle_features(
    opens: &[f32],
    closes: &[f32],
    highs: &[f32],
    lows: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut body = vec![0.0; opens.len()];
    let mut upper = vec![0.0; opens.len()];
    let mut lower = vec![0.0; opens.len()];

    for i in 0..opens.len() {
        body[i] = safe_div(closes[i] - opens[i], closes[i]);
        upper[i] = safe_div(highs[i] - opens[i].max(closes[i]), closes[i]);
        lower[i] = safe_div(opens[i].min(closes[i]) - lows[i], closes[i]);
    }

    (body, upper, lower)
}

fn price_z_score(closes: &[f32]) -> Vec<f32> {
    z_score(closes, 20)
}

fn normalized_range(highs: &[f32], lows: &[f32], closes: &[f32]) -> Vec<f32> {
    highs
        .iter()
        .zip(lows)
        .zip(closes)
        .map(|((h, l), c)| safe_div(h - l, *c))
        .collect()
}
