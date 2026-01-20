use crate::batcher::DataBatch;
use crate::consts::{CLASSES, TOTAL_FEATURE_SIZE};
use burn::Tensor;
use burn::config::Config;
use burn::module::Module;
use burn::nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig};
use burn::nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::record::CompactRecorder;
use burn::tensor::Int;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};
use std::path::Path;

/// The core machine learning model for Mirada AI.
#[derive(Debug, Module)]
pub struct Model<B: Backend> {
    loss: CrossEntropyLoss<B>,

    activation: Gelu,
    dropout: Dropout,

    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    linear4: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        let x = self.linear2.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        let x = self.linear3.forward(x);
        let x = self.activation.forward(x);

        self.linear4.forward(x)
    }

    pub fn forward_classification(
        &self,
        input: Tensor<B, 2>,
        target: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let out = self.forward(input);
        let loss = self.loss.forward(out.clone(), target.clone());
        ClassificationOutput::new(loss, out, target)
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

impl<B: AutodiffBackend> TrainStep<DataBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: DataBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.features, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<DataBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: DataBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.features, batch.targets)
    }
}

#[derive(Debug, Config)]
pub struct ModelConfig {
    #[config(default = 128)]
    pub hidden1: usize,
    #[config(default = 64)]
    pub hidden2: usize,
    #[config(default = 32)]
    pub hidden3: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            loss: CrossEntropyLossConfig {
                pad_tokens: None,
                weights: None,
                smoothing: None,
                logits: true,
            }
            .init(device),
            activation: Gelu,
            dropout: DropoutConfig::new(self.dropout).init(),
            linear1: LinearConfig::new(TOTAL_FEATURE_SIZE, self.hidden1).init(device),
            linear2: LinearConfig::new(self.hidden1, self.hidden2).init(device),
            linear3: LinearConfig::new(self.hidden2, self.hidden3).init(device),
            linear4: LinearConfig::new(self.hidden3, CLASSES).init(device),
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
