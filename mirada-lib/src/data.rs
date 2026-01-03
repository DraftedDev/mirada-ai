use crate::consts::{FEATURE_SIZE, HORIZON, OTHER_STOCKS};
use crate::math::{generate_targets, normalize, process};
use burn::Tensor;
use burn::prelude::Backend;
use burn::tensor::{Shape, TensorData};
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
    features: SerdeTensor<2>,
    targets: SerdeTensor<2>,
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
        let last_close = *closes.last().expect("No closes provided");

        let targets_data = if train {
            log::debug!("Generating targets data...");
            generate_targets(&closes, HORIZON)
        } else {
            log::debug!("Using empty targets data...");
            Vec::new()
        };

        log::debug!("Processing data into features...");
        let raw_features = process(opens, closes, volumes, highs, lows);

        log::debug!("Normalizing features data...");
        let norm_features = normalize(&raw_features);

        log::debug!("Finalizing features and targets data...");

        // Remove the last `HORIZON` rows during training to match target length.
        let features = if train {
            let slice_len = norm_features.len().saturating_sub(HORIZON);

            norm_features[..slice_len].to_vec()
        } else {
            norm_features.to_vec()
        };

        let flat_features: Vec<f32> = features.iter().flat_map(|x| x.iter()).copied().collect();

        let targets = if train {
            SerdeTensor::new([features.len(), 1], targets_data)
        } else {
            SerdeTensor::none()
        };

        Self {
            features: SerdeTensor::new([features.len(), FEATURE_SIZE], flat_features),
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
            features: SerdeTensor::from_tensor(Tensor::cat(merged, 1)),
            targets: self.targets,
            last_close: self.last_close,
        }
    }

    pub fn into_tensors<B: Backend>(self, device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>) {
        (
            self.features.to_tensor(device),
            self.targets.to_tensor(device),
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerdeTensor<const D: usize> {
    pub shape: Shape,
    pub data: Vec<f32>,
}

impl<const D: usize> SerdeTensor<D> {
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
