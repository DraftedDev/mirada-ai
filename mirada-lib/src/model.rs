use crate::batcher::DataBatch;
use crate::consts::{CLASSES, TOTAL_FEATURE_SIZE};
use crate::output::ModelOutput;
use burn::Tensor;
use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig};
use burn::nn::pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig};
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig, PaddingConfig1d};
use burn::prelude::Backend;
use burn::record::CompactRecorder;
use burn::tensor::Int;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{InferenceStep, TrainOutput, TrainStep};
use std::path::Path;

/// The core machine learning model for Mirada AI.
#[derive(Debug, Module)]
pub struct Model<B: Backend> {
    loss: CrossEntropyLoss<B>,

    conv: Conv1d<B>,
    conv_act: Gelu,
    dropout: Dropout,
    transformer: TransformerEncoder<B>,
    pool: AdaptiveAvgPool1d,
    head: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        // x: [B, T, F] -> [B, F, T]
        let x = x.swap_dims(1, 2);

        // Conv feature extraction
        let x = self.conv.forward(x);
        let x = self.conv_act.forward(x);
        let x = self.dropout.forward(x);

        // Transformer expects [B, T, C]
        let x = x.swap_dims(1, 2);

        let x = self.transformer.forward(TransformerEncoderInput::new(x));

        // back to [B, C, T]
        let x = x.swap_dims(1, 2);

        // pooling over time → [B, C, 1]
        let x = self.pool.forward(x);

        let x = x.squeeze::<2>(); // [B, C]

        // classification head
        self.head.forward(x)
    }

    pub fn forward_classification(
        &self,
        input: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
        returns: Tensor<B, 1>,
    ) -> ModelOutput<B> {
        let output = self.forward(input);
        let loss = self.loss.forward(output.clone(), targets.clone());

        ModelOutput {
            loss,
            output,
            targets,
            returns,
        }
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

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = DataBatch<B>;
    type Output = ModelOutput<B>;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let item = self.forward_classification(batch.features, batch.targets, batch.returns);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for Model<B> {
    type Input = DataBatch<B>;
    type Output = ModelOutput<B>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        self.forward_classification(batch.features, batch.targets, batch.returns)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 0.05)]
    pub loss_smoothing: f32,
    #[config(default = 0.1)]
    pub dropout: f64,
    #[config(default = 128)]
    pub d_model: usize,
    #[config(default = 256)]
    pub d_ff: usize,
    #[config(default = 4)]
    pub n_heads: usize,
    #[config(default = 2)]
    pub n_layers: usize,
    #[config(default = 3)]
    pub kernel_size: usize,
    #[config(default = 1)]
    pub pool_size: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            loss: CrossEntropyLossConfig::new()
                .with_logits(true)
                .with_smoothing(Some(self.loss_smoothing))
                .init(device),
            conv: Conv1dConfig::new(TOTAL_FEATURE_SIZE, self.d_model, self.kernel_size)
                .with_padding(PaddingConfig1d::Same)
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            conv_act: Gelu::new(),
            transformer: TransformerEncoderConfig::new(
                self.d_model,
                self.d_ff,
                self.n_heads,
                self.n_layers,
            )
            .with_dropout(self.dropout)
            .init(device),
            pool: AdaptiveAvgPool1dConfig::new(self.pool_size).init(),
            head: LinearConfig::new(self.d_model, CLASSES).init(device),
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
