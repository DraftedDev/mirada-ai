use crate::consts::{
    FEATURE_SIZE, HORIZON, OTHER_STOCKS, ROLLING_WINDOW, SKIPPED_TIMESTEPS, TEMP_WINDOWS,
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

        // Validate all OHLCV inputs
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

        // Minimum history needed
        let skip = ROLLING_WINDOW.max(SKIPPED_TIMESTEPS);
        assert!(
            n > skip + HORIZON + TEMP_WINDOWS - 1,
            "Not enough timesteps: need > skip({}) + horizon({}) + TEMP_WINDOWS({}), got {}",
            skip,
            HORIZON,
            TEMP_WINDOWS,
            n
        );

        // Cutoff for features (causal)
        let feature_end = if train { n - HORIZON } else { n };

        // Process & normalize features
        let mut features = process(
            &opens[..feature_end],
            &closes[..feature_end],
            &volumes[..feature_end],
            &highs[..feature_end],
            &lows[..feature_end],
        );
        features = normalize(&features);

        // Keep only timesteps after skip
        let features = &features[skip..feature_end];
        let n_days = features.len();
        assert!(
            n_days >= TEMP_WINDOWS,
            "Not enough timesteps ({n_days}) after skip for TEMP_WINDOWS ({TEMP_WINDOWS})"
        );

        let n_samples = n_days - TEMP_WINDOWS + 1;

        // Build temporal windows (3D)
        let mut windows: Vec<f32> = Vec::with_capacity(n_samples * TEMP_WINDOWS * FEATURE_SIZE);
        for t in 0..n_samples {
            for w in 0..TEMP_WINDOWS {
                windows.extend_from_slice(&features[t + w]);
            }
        }

        // Generate targets aligned with the **end of each temporal window**
        let targets = if train {
            let all_targets = generate_targets(&closes, HORIZON);

            // The first usable feature row corresponds to index `skip` in closes
            let first_feature_idx = skip;
            let targets_data: Vec<i32> = (0..n_samples)
                .map(|i| all_targets[first_feature_idx + i + TEMP_WINDOWS - 1])
                .collect();

            // Log target distribution
            let up_count = targets_data.iter().filter(|&&t| t == 1).count();
            let down_count = targets_data.len() - up_count;
            log::info!(
                "Data Targets: Up ({}), Down ({}) Ratio: ({:.2})",
                up_count,
                down_count,
                up_count as f32 / down_count as f32
            );

            IntSerdeTensor::new([n_samples], targets_data)
        } else {
            IntSerdeTensor::none()
        };

        log::info!(
            "Final dataset: {} temporal windows, each with {} timesteps and {} features (train={})",
            n_samples,
            TEMP_WINDOWS,
            FEATURE_SIZE,
            train
        );

        let feature_width = features[0].len();
        let expected_len = n_samples * TEMP_WINDOWS * feature_width;
        assert_eq!(
            windows.len(),
            expected_len,
            "Window data size mismatch: got {}, expected {}",
            windows.len(),
            expected_len
        );

        Self {
            features: FloatSerdeTensor::new([n_samples, TEMP_WINDOWS, feature_width], windows),
            targets,
            time,
        }
    }

    /// Merges the **features** from `other` into this [StockData].
    ///
    /// Does not merge targets and time since it wouldn't make sense to do so.
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

        let shape = self.features.shape.clone();

        // Convert self features to tensor
        let mut merged = vec![self.features.to_tensor::<B>(device)];

        // Convert other stocks' features to tensors and collect
        for data in others.into_iter() {
            assert_eq!(
                shape[0], data.features.shape[0],
                "All stocks must have the same number of timesteps"
            );
            let t = data.features.to_tensor::<B>(device);
            assert_eq!(
                t.dims().len(),
                3,
                "Expected 3D tensor for temporal windows [n_samples, TEMP_WINDOWS, FEATURE_SIZE]"
            );
            merged.push(t);
        }

        // Concatenate along the feature dimension (last axis)
        let cat_features = Tensor::cat(merged, 2); // dim=2 = feature axis

        Self {
            features: FloatSerdeTensor::from_tensor(cat_features),
            targets: self.targets,
            time: self.time,
        }
    }

    pub fn into_tensors<B: Backend>(self, device: &B::Device) -> (Tensor<B, 3>, Tensor<B, 1, Int>) {
        (
            self.features.to_tensor(device),
            self.targets.to_tensor(device),
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
