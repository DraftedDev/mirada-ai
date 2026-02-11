use crate::batcher::DataBatch;
use crate::consts::TOTAL_FEATURE_SIZE;
use burn::Tensor;
use burn::config::Config;
use burn::module::Module;
use burn::nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig};
use burn::nn::{
    Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig, SwiGlu,
    SwiGluConfig,
};
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
    act: Gelu,
    dropout: Dropout,
    lin1: Linear<B>,
    lin2: Linear<B>,
    glu: SwiGlu<B>,
    norm: LayerNorm<B>,
    lin3: Linear<B>,
    lin4: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = self.lin1.forward(input);
        x = self.act.forward(x);
        x = self.dropout.forward(x);
        x = self.norm.forward(x);
        x = self.lin2.forward(x);
        x = self.glu.forward(x);
        x = self.dropout.forward(x);
        x = self.lin3.forward(x);
        x = self.act.forward(x);
        x = self.dropout.forward(x);
        self.lin4.forward(x)
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

#[derive(Config, Debug)]
pub struct ModelConfig {
    /// Loss smoothing factor.
    #[config(default = 0.05)]
    pub loss_smoothing: f32,

    /// Hidden dimension for Linear and SwiGLU layers.
    #[config(default = 128)]
    pub hidden_dim: usize,

    /// Dropout probability.
    #[config(default = 0.2)]
    pub dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            loss: CrossEntropyLossConfig::new()
                .with_logits(true)
                .with_smoothing(Some(self.loss_smoothing))
                .init(device),
            act: Gelu::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
            lin1: LinearConfig::new(TOTAL_FEATURE_SIZE, self.hidden_dim).init(device),
            lin2: LinearConfig::new(self.hidden_dim, self.hidden_dim * 2).init(device), // expand to hidden_dim * 2
            glu: SwiGluConfig::new(self.hidden_dim * 2, self.hidden_dim).init(device), // compress back to hidden_dim
            norm: LayerNormConfig::new(self.hidden_dim).init(device),
            lin3: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            lin4: LinearConfig::new(self.hidden_dim, 2).init(device), // 2 classes: up/down
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
