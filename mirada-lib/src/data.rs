use crate::math::{generate_targets, normalize, process};
use burn::Tensor;
use burn::prelude::Backend;
use burn::tensor::{Shape, TensorData};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use time::OffsetDateTime;

pub const WINDOW_Z: usize = 90;
pub const WINDOW_SCALE: usize = 252;
pub const CLIP: f32 = 6.0;
pub const HORIZON: usize = 3;
pub const FEATURE_SIZE: usize = 18;
pub const OTHER_STOCKS: usize = 5;
pub const TOTAL_FEATURE_SIZE: usize = FEATURE_SIZE * (OTHER_STOCKS + 1);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataKey {
    ticker: String,
    start: OffsetDateTime,
    end: OffsetDateTime,
    interval: String,
}

impl DataKey {
    pub fn new(
        ticker: String,
        start: OffsetDateTime,
        end: OffsetDateTime,
        interval: String,
    ) -> Self {
        Self {
            ticker,
            start,
            end,
            interval,
        }
    }
}

impl Display for DataKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ticker({}) start({}) end({}) interval({})",
            self.ticker,
            self.start.date(),
            self.end.date(),
            self.interval
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
    ) -> Self {
        let last_close = *closes.last().expect("No closes provided");

        log::info!("Generating targets data...");
        let targets = generate_targets(&closes, HORIZON);

        log::info!("Processing data into features...");
        let raw_features = process(opens, closes, volumes, highs, lows);

        log::info!("Normalizing features data...");
        let norm_features = normalize(&raw_features, WINDOW_Z, WINDOW_SCALE, CLIP);

        log::info!("Finalizing features and targets data...");
        // Remove last `horizon` rows from features to match target length
        let aligned_features: Vec<[f32; FEATURE_SIZE]> =
            norm_features[..norm_features.len() - HORIZON].to_vec();

        let flat_features: Vec<f32> = aligned_features
            .iter()
            .flat_map(|x| x.iter())
            .copied()
            .collect();

        Self {
            features: SerdeTensor::new([aligned_features.len(), FEATURE_SIZE], flat_features),
            targets: SerdeTensor::new([aligned_features.len(), 1], targets),
            last_close,
        }
    }

    pub fn last_close(&self) -> f32 {
        self.last_close
    }

    /// Merges the **features** from `other` into this [StockData].
    ///
    /// Does not merge targets or last closes, since it wouldn't make sense to do so.
    pub fn merge<B: Backend>(self, others: Vec<StockData>, device: &B::Device) -> Self {
        assert_eq!(
            others.len(),
            OTHER_STOCKS,
            "Can only process {OTHER_STOCKS} other stocks"
        );

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
