use crate::batcher::DataBatch;
use crate::data::TOTAL_FEATURE_SIZE;
use burn::Tensor;
use burn::config::Config;
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{RegressionOutput, TrainOutput, TrainStep, ValidStep};
use std::path::Path;

#[derive(Debug, Module)]
pub struct Model<B: Backend> {
    activation: Gelu,
    linear1: Linear<B>,
    // activation
    norm1: LayerNorm<B>,
    linear2: Linear<B>,
    // activation
    norm2: LayerNorm<B>,
    linear3: Linear<B>,
    // activation
    dropout: Dropout,
    head: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let linear1 = self.linear1.forward(input);
        let activation1 = self.activation.forward(linear1);
        let norm1 = self.norm1.forward(activation1);
        let linear2 = self.linear2.forward(norm1);
        let activation2 = self.activation.forward(linear2);
        let norm2 = self.norm2.forward(activation2);
        let linear3 = self.linear3.forward(norm2);
        let activation3 = self.activation.forward(linear3);
        let dropout = self.dropout.forward(activation3);

        self.head.forward(dropout)
    }

    pub fn forward_regression(
        &self,
        input: Tensor<B, 2>,
        target: Tensor<B, 2>,
    ) -> RegressionOutput<B> {
        let out = self.forward(input);

        let loss = MseLoss.forward(out.clone(), target.clone(), Reduction::Mean);

        RegressionOutput::new(loss, out, target)
    }

    pub fn load(config: ModelConfig, path: impl AsRef<Path>, device: &B::Device) -> Self {
        let recorder = Self::recorder();
        let model = config.init(device);

        model
            .load_file(path.as_ref(), &recorder, device)
            .expect("Failed to load model")
    }

    pub fn recorder() -> CompactRecorder {
        CompactRecorder::new()
    }
}

impl<B: AutodiffBackend> TrainStep<DataBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: DataBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(batch.features, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<DataBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: DataBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(batch.features, batch.targets)
    }
}

#[derive(Debug, Config)]
pub struct ModelConfig {
    /// Number of hidden units per hidden layer.
    #[config(default = 64)]
    pub hidden_size: usize,

    /// Dropout probability for the hidden layer.
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl ModelConfig {
    /// Initialize the MLP model from this config
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            activation: Gelu,
            linear1: LinearConfig::new(TOTAL_FEATURE_SIZE, self.hidden_size).init(device),
            norm1: LayerNormConfig::new(self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            norm2: LayerNormConfig::new(self.hidden_size).init(device),
            linear3: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            head: LinearConfig::new(self.hidden_size, 1).init(device),
        }
    }

    pub fn load_or_create(path: impl AsRef<Path>) -> ModelConfig {
        let path = path.as_ref();

        log::info!("Loading model config from '{}'...", path.display());

        if path.exists() {
            ModelConfig::load(path).expect("Failed to load model config")
        } else {
            let default = ModelConfig::new();

            default.save(path).expect("Failed to save model config");

            default
        }
    }
}
