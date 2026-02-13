use crate::batcher::DataBatch;
use crate::consts::TOTAL_FEATURE_SIZE;
use burn::Tensor;
use burn::config::Config;
use burn::module::Module;
use burn::nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig};
use burn::nn::{
    Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig, Lstm,
    LstmConfig,
};
use burn::prelude::Backend;
use burn::record::CompactRecorder;
use burn::tensor::Int;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep};
use std::path::Path;

/// The core machine learning model for Mirada AI.
#[derive(Debug, Module)]
pub struct Model<B: Backend> {
    loss: CrossEntropyLoss<B>,
    lstm1: Lstm<B>,
    lstm2: Lstm<B>,
    dropout: Dropout,
    norm: LayerNorm<B>,
    act: Gelu,
    lin: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let (x, _) = self.lstm1.forward(x, None);
        let (x, state2) = self.lstm2.forward(x, None);

        // Mean over timesteps -> result likely has shape [batch, 1, hidden], so squeeze the middle dim
        let x_mean = x.mean_dim(1).squeeze_dim::<2>(1); // -> [batch, hidden]

        // state2.hidden is already [batch, hidden]
        let last_layer_h = state2.hidden; // -> [batch, hidden]

        // Concatenate along feature axis -> [batch, hidden*2]
        let x = Tensor::cat(vec![last_layer_h, x_mean], 1);

        let x = self.dropout.forward(x);
        let x = self.norm.forward(x);
        let x = self.act.forward(x);
        self.lin.forward(x)
    }

    pub fn forward_classification(
        &self,
        input: Tensor<B, 3>,
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

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = DataBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: DataBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.features, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for Model<B> {
    type Input = DataBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: DataBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.features, batch.targets)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    /// Loss smoothing factor.
    #[config(default = 0.05)]
    pub loss_smoothing: f32,

    /// Hidden size of layers like Linear and Lstm.
    #[config(default = 128)]
    pub hidden_size: usize,

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
            lstm1: LstmConfig::new(TOTAL_FEATURE_SIZE, self.hidden_size, true).init(device),
            lstm2: LstmConfig::new(self.hidden_size, self.hidden_size, true).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            norm: LayerNormConfig::new(self.hidden_size * 2).init(device),
            act: Gelu,
            lin: LinearConfig::new(self.hidden_size * 2, 2).init(device),
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
