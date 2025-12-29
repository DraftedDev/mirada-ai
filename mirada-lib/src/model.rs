use crate::batcher::DataBatch;
use crate::data::TOTAL_FEATURE_SIZE;
use crate::residual::{ResidualBlock, ResidualBlockConfig};
use burn::Tensor;
use burn::config::Config;
use burn::module::Module;
use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{RegressionOutput, TrainOutput, TrainStep, ValidStep};
use std::path::Path;

/// The core machine learning model for Mirada AI.
///
/// It uses following layer architecture to predict:
/// 1. Input Linear Layer
#[derive(Debug, Module)]
pub struct Model<B: Backend> {
    input_linear: Linear<B>,
    input_activation: Gelu,
    input_norm: LayerNorm<B>,

    residual_blocks: Vec<ResidualBlock<B>>,

    attention: MultiHeadAttention<B>,
    attention_norm: LayerNorm<B>,

    mlp_linear: Linear<B>,
    mlp_activation: Gelu,
    mlp_dropout: Dropout,

    head: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = self.input_linear.forward(input);
        x = self.input_activation.forward(x);
        x = self.input_norm.forward(x);

        for block in &self.residual_blocks {
            x = block.forward(x);
        }

        let x_seq = x.clone().unsqueeze_dim::<3>(1);

        // TODO: make attention better (ask chat gpt)
        let attn_out = self.attention.forward(MhaInput::self_attn(x_seq));
        let x_attn = attn_out.context.squeeze_dim::<2>(1);
        x = self.attention_norm.forward(x_attn + x);

        x = self.mlp_linear.forward(x);
        x = self.mlp_activation.forward(x);
        x = self.mlp_dropout.forward(x);

        self.head.forward(x)
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
    #[config(default = 128)]
    pub hidden_dim: usize,
    #[config(default = 5)]
    pub residual_blocks: usize,
    #[config(default = 8)]
    pub n_heads: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            input_linear: LinearConfig::new(TOTAL_FEATURE_SIZE, self.hidden_dim).init(device),
            input_activation: Gelu::new(),
            input_norm: LayerNormConfig::new(self.hidden_dim).init(device),
            residual_blocks: vec![
                ResidualBlockConfig::new(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.dropout
                )
                .init(device);
                self.residual_blocks
            ],
            attention: MultiHeadAttentionConfig::new(self.hidden_dim, self.n_heads).init(device),
            attention_norm: LayerNormConfig::new(self.hidden_dim).init(device),
            mlp_linear: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            mlp_activation: Gelu::new(),
            mlp_dropout: DropoutConfig::new(self.dropout).init(),
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
