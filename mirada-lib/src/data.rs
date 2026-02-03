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
    last_close: f32,
}

impl StockData {
    pub fn new(
        opens: Vec<f32>,
        closes: Vec<f32>,
        volumes: Vec<f32>,
        highs: Vec<f32>,
        lows: Vec<f32>,
        train: bool,
    ) -> Self {
        let n = closes.len();
        let last_close = *closes.last().expect("No closes provided");

        let skip = ROLLING_WINDOW.max(SKIPPED_TIMESTEPS);

        assert!(
            n > skip + HORIZON,
            "Not enough timesteps: need > skip({}) + horizon({}), got {}",
            skip,
            HORIZON,
            n
        );

        // Total usable supervised samples
        let usable = n - skip - HORIZON;

        // -------- FEATURES --------
        log::debug!("Processing data into features...");
        let raw_features = process(&opens, &closes, &volumes, &highs, &lows);

        log::debug!("Normalizing features...");
        let norm_features = normalize(&raw_features); // length = n - skip

        assert!(
            norm_features.len() >= usable,
            "Normalized features too short: {} < usable {}",
            norm_features.len(),
            usable
        );

        // features[i] corresponds to time t = skip + i
        let features: Vec<[f32; FEATURE_SIZE]> = norm_features[..usable].to_vec();

        // -------- TARGETS --------
        let targets_data = if train {
            log::debug!("Generating targets...");

            let all_targets = generate_targets(&closes, HORIZON);
            // all_targets[t] corresponds to time t

            let targets = all_targets[skip..skip + usable].to_vec();

            assert_eq!(
                features.len(),
                targets.len(),
                "Feature/target misalignment: {} vs {}",
                features.len(),
                targets.len()
            );

            targets
        } else {
            Vec::new()
        };

        log::info!(
            "Final dataset: {} samples, {} features each",
            features.len(),
            FEATURE_SIZE
        );

        // Flatten features
        let flat_features: Vec<f32> = features.iter().flat_map(|x| x.iter()).copied().collect();

        let targets = if train {
            IntSerdeTensor::new([features.len()], targets_data)
        } else {
            IntSerdeTensor::none()
        };

        Self {
            features: FloatSerdeTensor::new([features.len(), FEATURE_SIZE], flat_features),
            targets,
            last_close,
        }
    }

    pub fn last_close(&self) -> f32 {
        self.last_close
    }

    /// Merges the **features** from `other` into this [StockData].
    ///
    /// Does not merge targets, last closes or last rolling volatility, since it wouldn't make sense to do so.
    pub fn merge<B: Backend>(self, others: Vec<StockData>, device: &B::Device) -> Self {
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
            last_close: self.last_close,
        }
    }

    pub fn into_tensors<B: Backend>(self, device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 1, Int>) {
        (
            self.features.to_tensor(device),
            self.targets.to_tensor(device),
        )
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
