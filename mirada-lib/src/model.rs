use crate::batcher::DataBatch;
use crate::consts::TOTAL_FEATURE_SIZE;
use crate::residual::{ResidualBlock, ResidualBlockConfig};
use burn::Tensor;
use burn::config::Config;
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{
    Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig, Sigmoid,
};
use burn::prelude::Backend;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{RegressionOutput, TrainOutput, TrainStep, ValidStep};
use std::path::Path;

/// The core machine learning model for Mirada AI.
#[derive(Debug, Module)]
pub struct Model<B: Backend> {
    feature_gate: Linear<B>,
    sigmoid: Sigmoid,
    input_linear: Linear<B>,
    input_norm: LayerNorm<B>,
    input_activation: Gelu,

    residual_blocks: Vec<ResidualBlock<B>>,

    mlp_linear: Linear<B>,
    mlp_activation: Gelu,
    mlp_dropout: Dropout,

    head_norm: LayerNorm<B>,
    head: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // Feature gating
        let mut x = {
            let gate = self.feature_gate.forward(input.clone());

            self.sigmoid.forward(gate) * input
        };

        // Input linear + norm + activation
        x = {
            let lin = self.input_linear.forward(x);
            let norm = self.input_norm.forward(lin);

            self.input_activation.forward(norm)
        };

        // Residual blocks
        for block in &self.residual_blocks {
            x = block.forward(x);
        }

        // MLP Linear + activation + dropout + residual
        x = {
            let mlp_residual = x.clone();

            x = self.mlp_linear.forward(x);
            x = self.mlp_activation.forward(x);
            x = self.mlp_dropout.forward(x);

            x + mlp_residual
        };

        // Head Norm + Linear
        {
            x = self.head_norm.forward(x);
            self.head.forward(x)
        }
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
    #[config(default = 256)]
    pub hidden_dim: usize,
    #[config(default = 8)]
    pub residual_blocks: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            feature_gate: LinearConfig::new(TOTAL_FEATURE_SIZE, TOTAL_FEATURE_SIZE).init(device),
            sigmoid: Sigmoid::new(),
            input_linear: LinearConfig::new(TOTAL_FEATURE_SIZE, self.hidden_dim).init(device),
            input_activation: Gelu::new(),
            input_norm: LayerNormConfig::new(self.hidden_dim)
                .with_epsilon(1e-5)
                .init(device),
            residual_blocks: vec![
                ResidualBlockConfig::new(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.dropout
                )
                .init(device);
                self.residual_blocks
            ],
            mlp_linear: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            mlp_activation: Gelu::new(),
            mlp_dropout: DropoutConfig::new(self.dropout).init(),
            head_norm: LayerNormConfig::new(self.hidden_dim).init(device),
            head: LinearConfig::new(self.hidden_dim, 1).init(device),
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
