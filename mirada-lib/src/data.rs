use crate::consts::{
    EPS, HORIZON, OTHER_STOCKS, ROLLING_WINDOW, SKIPPED_TIMESTEPS, TEMP_WINDOWS, TOTAL_FEATURE_SIZE,
};
use crate::math::{generate_targets, normalize, process};
use burn::Tensor;
use burn::prelude::Backend;
use burn::tensor::{Int, Shape, TensorData};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use time::OffsetDateTime;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataKey {
    ticker: String,
    start: OffsetDateTime,
    end: OffsetDateTime,
}

impl DataKey {
    pub fn new(ticker: String, start: OffsetDateTime, end: OffsetDateTime) -> Self {
        Self { ticker, start, end }
    }
}

impl Display for DataKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ticker({}) start({}) end({})",
            self.ticker,
            self.start.date(),
            self.end.date(),
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StockData {
    features: FloatSerdeTensor<3>,
    targets: IntSerdeTensor<1>,
    returns: FloatSerdeTensor<1>,
    time: (OffsetDateTime, OffsetDateTime),
}

impl StockData {
    pub fn new(
        opens: Vec<f32>,
        closes: Vec<f32>,
        closes_other: [Vec<f32>; OTHER_STOCKS],
        volumes: Vec<f32>,
        highs: Vec<f32>,
        lows: Vec<f32>,
        time: (OffsetDateTime, OffsetDateTime),
        train: bool,
    ) -> Self {
        let n = closes.len();
        assert!(n > 0, "Empty input series");

        // Validate primary OHLCV inputs.
        assert!(
            opens.len() == n && volumes.len() == n && highs.len() == n && lows.len() == n,
            "OHLCV series must already be date-aligned"
        );

        for i in 0..n {
            assert!(opens[i].is_finite(), "Open data [{i}] must be finite");
            assert!(closes[i].is_finite(), "Close data [{i}] must be finite");
            assert!(volumes[i].is_finite(), "Volume data [{i}] must be finite");
            assert!(highs[i].is_finite(), "High data [{i}] must be finite");
            assert!(lows[i].is_finite(), "Low data [{i}] must be finite");
        }

        // Validate cross-asset closes too.
        for (k, series) in closes_other.iter().enumerate() {
            assert_eq!(
                series.len(),
                n,
                "closes_other[{k}] must be date-aligned with the main series"
            );
            for (i, v) in series.iter().enumerate() {
                assert!(v.is_finite(), "closes_other[{k}][{i}] must be finite");
            }
        }

        // Minimum history needed.
        let skip = ROLLING_WINDOW.max(SKIPPED_TIMESTEPS);
        assert!(
            n > skip + HORIZON + TEMP_WINDOWS - 1,
            "Not enough timesteps: need > skip({}) + horizon({}) + TEMP_WINDOWS({}), got {}",
            skip,
            HORIZON,
            TEMP_WINDOWS,
            n
        );

        // Cutoff for features (causal).
        let feature_end = if train { n - HORIZON } else { n };

        // Build features.
        let mut features = process(
            opens[..feature_end].to_vec(),
            closes[..feature_end].to_vec(),
            closes_other
                .iter()
                .map(|v| v[..feature_end].to_vec())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            volumes[..feature_end].to_vec(),
            highs[..feature_end].to_vec(),
            lows[..feature_end].to_vec(),
        );
        features = normalize(features);

        assert!(!features.is_empty(), "process() produced no features");

        let feature_width = features[0].len();
        assert_eq!(
            feature_width, TOTAL_FEATURE_SIZE,
            "FEATURE_SIZE mismatch: expected {}, got {}",
            TOTAL_FEATURE_SIZE, feature_width
        );

        // Keep only usable timesteps.
        let features = &features[skip..];
        let n_days = features.len();
        assert!(
            n_days >= TEMP_WINDOWS,
            "Not enough timesteps ({n_days}) after skip for TEMP_WINDOWS ({TEMP_WINDOWS})"
        );

        let n_samples = n_days - TEMP_WINDOWS + 1;

        // Build temporal windows: [sample, time, feature].
        let mut windows: Vec<f32> = Vec::with_capacity(n_samples * TEMP_WINDOWS * feature_width);
        for t in 0..n_samples {
            for w in 0..TEMP_WINDOWS {
                windows.extend_from_slice(&features[t + w]);
            }
        }

        // Targets aligned with the end of each temporal window.
        let targets = if train {
            let all_targets = generate_targets(&closes, HORIZON);
            let first_feature_idx = skip;

            let targets_data: Vec<i32> = (0..n_samples)
                .map(|i| all_targets[first_feature_idx + i + TEMP_WINDOWS - 1])
                .collect();

            let up_count = targets_data.iter().filter(|&&t| t == 2).count();
            let down_count = targets_data.iter().filter(|&&t| t == 0).count();
            let denom = down_count.max(1) as f32;

            log::info!(
                "Data Targets: Up ({}), Down ({}) Ratio: ({:.2})",
                up_count,
                down_count,
                up_count as f32 / denom
            );

            IntSerdeTensor::new([n_samples], targets_data)
        } else {
            IntSerdeTensor::none()
        };

        log::info!(
            "Final dataset: {} temporal windows, each with {} timesteps and {} features (train={})",
            n_samples,
            TEMP_WINDOWS,
            feature_width,
            train
        );

        let returns = if train {
            let mut ret = Vec::with_capacity(n_samples);
            let first_feature_idx = skip;

            for i in 0..n_samples {
                let t = first_feature_idx + i + TEMP_WINDOWS - 1;

                let c0 = closes[t].max(EPS);
                let c1 = closes[t + HORIZON].max(EPS);

                ret.push((c1 / c0).ln());
            }

            FloatSerdeTensor::new([n_samples], ret)
        } else {
            FloatSerdeTensor::none()
        };

        Self {
            features: FloatSerdeTensor::new([n_samples, TEMP_WINDOWS, feature_width], windows),
            targets,
            returns,
            time,
        }
    }

    pub fn into_tensors<B: Backend>(
        self,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 1, Int>, Tensor<B, 1>) {
        (
            self.features.to_tensor(device),
            self.targets.to_tensor(device),
            self.returns.to_tensor(device),
        )
    }

    pub fn features(&self) -> &FloatSerdeTensor<3> {
        &self.features
    }

    pub fn targets(&self) -> &IntSerdeTensor<1> {
        &self.targets
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FloatSerdeTensor<const D: usize> {
    pub shape: Shape,
    pub data: Vec<f32>,
}

impl<const D: usize> FloatSerdeTensor<D> {
    pub fn new(shape: impl Into<Shape>, data: Vec<f32>) -> Self {
        Self {
            shape: shape.into(),
            data,
        }
    }

    pub fn none() -> Self {
        Self::new([0; D], Vec::new())
    }

    pub fn to_tensor<B: Backend>(self, device: &B::Device) -> Tensor<B, D> {
        Tensor::from_data(TensorData::new(self.data, self.shape), device)
    }

    pub fn from_tensor<B: Backend>(tensor: Tensor<B, D>) -> Self {
        let shape = tensor.shape();
        let data = tensor
            .to_data()
            .to_vec::<f32>()
            .expect("Failed to convert tensor to vector");

        Self { shape, data }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntSerdeTensor<const D: usize> {
    pub shape: Shape,
    pub data: Vec<i32>,
}

impl<const D: usize> IntSerdeTensor<D> {
    pub fn new(shape: impl Into<Shape>, data: Vec<i32>) -> Self {
        Self {
            shape: shape.into(),
            data,
        }
    }

    pub fn none() -> Self {
        Self::new([0; D], Vec::new())
    }

    pub fn to_tensor<B: Backend>(self, device: &B::Device) -> Tensor<B, D, Int> {
        Tensor::from_data(TensorData::new(self.data, self.shape), device)
    }

    pub fn from_tensor<B: Backend>(tensor: Tensor<B, D, Int>) -> Self {
        let shape = tensor.shape();
        let data = tensor
            .to_data()
            .to_vec::<i32>()
            .expect("Failed to convert tensor to vector");

        Self { shape, data }
    }
}
