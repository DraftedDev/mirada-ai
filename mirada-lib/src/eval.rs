use crate::batcher::StockBatcher;
use crate::database::Database;
use crate::dataset::StockDataset;
use crate::metrics::SharpeRatioMetrics;
use crate::model::Model;
use burn::config::Config;
use burn::data::dataloader::{DataLoaderBuilder, Dataset};
use burn::tensor::backend::AutodiffBackend;
use burn::train::EvaluatorBuilder;
use burn::train::metric::{AccuracyMetric, ClassReduction, LossMetric, PrecisionMetric};
use std::path::Path;

impl<B: AutodiffBackend> Model<B> {
    pub fn eval(
        &self,
        config: EvalConfig,
        database: Database,
        dataset: StockDataset,
        artifacts: impl AsRef<Path>,
        device: B::Device,
    ) {
        B::seed(&device, config.seed);

        let batcher = StockBatcher::new(database.clone());

        assert!(!dataset.is_empty(), "Datasets must not be empty");

        let dataloader = DataLoaderBuilder::new(batcher.clone())
            .batch_size(config.batch_size)
            .num_workers(config.num_workers)
            .set_device(device)
            .build(dataset);

        let eval = EvaluatorBuilder::new(artifacts)
            .metric_numeric(SharpeRatioMetrics::default())
            .metric_numeric(PrecisionMetric::multiclass(2, ClassReduction::Macro))
            .metric_numeric(AccuracyMetric::new())
            .metric_numeric(LossMetric::new())
            .summary()
            .build(self.clone());

        eval.eval("Evaluation", dataloader);
    }
}

#[derive(Config, Debug)]
pub struct EvalConfig {
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 12)]
    pub seed: u64,
}

impl EvalConfig {
    pub fn load_or_create(path: impl AsRef<Path>) -> EvalConfig {
        let path = path.as_ref();

        log::info!("Loading eval config from '{}'...", path.display());

        if path.exists() {
            EvalConfig::load(path).expect("Failed to load eval config")
        } else {
            let default = EvalConfig::new();

            default.save(path).expect("Failed to save eval config");

            default
        }
    }
}
