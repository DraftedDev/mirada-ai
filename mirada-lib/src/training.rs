use crate::batcher::StockBatcher;
use crate::database::Database;
use crate::dataset::StockDataset;
use crate::model::Model;
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::module::Module;
use burn::optim::AdamWConfig;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{AccuracyMetric, LearningRateMetric, LossMetric, PrecisionMetric};
use burn::train::{LearnerBuilder, LearningStrategy};
use std::path::Path;

impl<B: AutodiffBackend> Model<B> {
    pub fn train(
        &self,
        config: TrainingConfig,
        database: Database,
        train_set: StockDataset,
        valid_set: StockDataset,
        artifacts: impl AsRef<Path>,
        device: B::Device,
    ) {
        let artifacts = artifacts.as_ref();

        B::seed(&device, config.seed);

        let batcher = StockBatcher::new(database.clone());

        let train_dataloader = DataLoaderBuilder::new(batcher.clone())
            .batch_size(config.batch_size)
            .num_workers(config.num_workers)
            .shuffle(config.seed)
            .build(train_set);

        let valid_dataloader = DataLoaderBuilder::new(batcher.clone())
            .batch_size(config.batch_size)
            .num_workers(config.num_workers)
            .build(valid_set);

        log::info!("Initializing optimizer...");
        let optimizer = AdamWConfig::new()
            .with_weight_decay(config.weight_decay)
            .init();

        let recorder = Self::recorder();

        log::info!("Initializing learner...");
        let learner = LearnerBuilder::new(artifacts)
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .metric_train_numeric(LearningRateMetric::new())
            .metric_valid_numeric(LearningRateMetric::new())
            .metric_train_numeric(AccuracyMetric::new())
            .metric_valid_numeric(AccuracyMetric::new())
            .metric_train_numeric(PrecisionMetric::binary(0.5))
            .metric_valid_numeric(PrecisionMetric::binary(0.5))
            .with_file_checkpointer(recorder.clone())
            .learning_strategy(LearningStrategy::SingleDevice(device))
            .num_epochs(config.num_epochs)
            .summary()
            .build(
                self.clone(),
                optimizer,
                CosineAnnealingLrSchedulerConfig::new(config.init_learning_rate, config.num_epochs)
                    .with_min_lr(config.min_learning_rate)
                    .init()
                    .expect("Failed to initialize learning rate scheduler"),
            );

        let result = learner.fit(train_dataloader, valid_dataloader);

        result
            .model
            .save_file(artifacts.join("model"), &recorder)
            .expect("Failed to save model");
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = 1e-4)]
    pub weight_decay: f32,
    #[config(default = 100)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 12)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub init_learning_rate: f64,
    #[config(default = 1.0e-5)]
    pub min_learning_rate: f64,
}

impl TrainingConfig {
    pub fn load_or_create(path: impl AsRef<Path>) -> TrainingConfig {
        let path = path.as_ref();

        log::info!("Loading training config from '{}'...", path.display());

        if path.exists() {
            TrainingConfig::load(path).expect("Failed to load training config")
        } else {
            let default = TrainingConfig::new();

            default.save(path).expect("Failed to save training config");

            default
        }
    }
}
