use crate::consts::{FEATURE_SIZE, HORIZON, OTHER_STOCKS, ROLLING_WINDOW, SKIPPED_TIMESTEPS};
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
    features: FloatSerdeTensor<2>,
    targets: IntSerdeTensor<1>,
    time: (OffsetDateTime, OffsetDateTime),
}

impl StockData {
    pub fn new(
        opens: Vec<f32>,
        closes: Vec<f32>,
        volumes: Vec<f32>,
        highs: Vec<f32>,
        lows: Vec<f32>,
        time: (OffsetDateTime, OffsetDateTime),
        train: bool,
    ) -> Self {
        let n = closes.len();
        assert!(n > 0, "Empty input series");

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

        // minimum amount of history needed before producing valid features
        let skip = ROLLING_WINDOW.max(SKIPPED_TIMESTEPS);

        assert!(
            n > skip + HORIZON,
            "Not enough timesteps: need > skip({}) + horizon({}), got {}",
            skip,
            HORIZON,
            n
        );

        // make features only see past data (if training)
        let feature_end = if train { n - HORIZON } else { n };

        let raw_features = process(
            &opens[..feature_end],
            &closes[..feature_end],
            &volumes[..feature_end],
            &highs[..feature_end],
            &lows[..feature_end],
        );

        assert_eq!(
            raw_features.len(),
            feature_end,
            "Feature pipeline changed series length → misalignment risk"
        );

        let norm_features = normalize(&raw_features);

        // only keep timesteps after initial skip to have enough history
        let features: Vec<[f32; FEATURE_SIZE]> = norm_features[skip..].to_vec();

        let targets = if train {
            let all_targets = generate_targets(&closes, HORIZON);

            // only take targets aligned with features (skip first `skip` timesteps)
            let targets_data = all_targets[skip..feature_end].to_vec();

            assert_eq!(
                features.len(),
                targets_data.len(),
                "Features and targets must have the same size"
            );

            IntSerdeTensor::new([features.len()], targets_data)
        } else {
            IntSerdeTensor::none()
        };

        // flatten features for storage
        let flat_features: Vec<f32> = features
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .collect();

        log::info!(
            "Final dataset: {} samples with {} features (train={})",
            features.len(),
            FEATURE_SIZE,
            train
        );

        Self {
            features: FloatSerdeTensor::new([features.len(), FEATURE_SIZE], flat_features),
            targets,
            time,
        }
    }

    /// Merges the **features** from `other` into this [StockData].
    ///
    /// Does not merge targets, last closes or last rolling volatility, since it wouldn't make sense to do so.
    pub fn merge<B: Backend>(self, others: Vec<StockData>, device: &B::Device) -> Self {
        for other in &others {
            assert_eq!(
                self.time, other.time,
                "All stocks must have the same start and end date"
            );
        }

        assert_eq!(
            others.len(),
            OTHER_STOCKS,
            "Can only process {OTHER_STOCKS} other stocks"
        );

        if self.features.shape[0] == 0 {
            panic!("Stock features cannot have zero size");
        } else if others.iter().any(|d| d.features.shape[0] == 0) {
            panic!("Other stocks features cannot have zero size");
        }

        let mut merged = Vec::with_capacity(others.len() + 1);

        let other_features = others
            .into_iter()
            .map(|data| {
                assert_eq!(
                    self.features.shape[0], data.features.shape[0],
                    "All stocks must have the same number of timesteps"
                );

                data.features.to_tensor::<B>(device)
            })
            .collect::<Vec<_>>();

        merged.push(self.features.to_tensor(device));
        merged.extend(other_features);

        assert!(
            merged.iter().all(|t| t.dims()[1] > 0),
            "Attempted to merge a tensor with zero size"
        );

        Self {
            features: FloatSerdeTensor::from_tensor(Tensor::cat(merged, 1)),
            targets: self.targets,
            time: self.time,
        }
    }

    pub fn into_tensors<B: Backend>(self, device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 1, Int>) {
        (
            self.features.to_tensor(device),
            self.targets.to_tensor(device),
        )
    }

    pub fn features(&self) -> &FloatSerdeTensor<2> {
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
