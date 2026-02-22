use crate::batcher::StockBatcher;
use crate::database::Database;
use crate::dataset::StockDataset;
use crate::metrics::SharpeRatioMetrics;
use crate::model::Model;
use crate::training::TrainingConfig;
use burn::data::dataloader::{DataLoaderBuilder, Dataset};
use burn::tensor::backend::AutodiffBackend;
use burn::train::EvaluatorBuilder;
use burn::train::metric::{AccuracyMetric, ClassReduction, LossMetric, PrecisionMetric};
use std::path::Path;

impl<B: AutodiffBackend> Model<B> {
    pub fn eval(
        &self,
        config: TrainingConfig,
        database: Database,
        dataset: StockDataset,
        artifacts: impl AsRef<Path>,
        device: B::Device,
    ) {
        B::seed(&device, config.seed);

        let batcher = StockBatcher::new(database.clone());

        assert!(!dataset.is_empty(), "Datasets must not be empty");

        let dataloader = DataLoaderBuilder::new(batcher.clone())
            .batch_size(1)
            .num_workers(config.num_workers)
            .set_device(device)
            .build(dataset);

        let eval = EvaluatorBuilder::new(artifacts)
            .metric_numeric(SharpeRatioMetrics::default())
            .metric_numeric(PrecisionMetric::multiclass(1, ClassReduction::Macro))
            .metric_numeric(AccuracyMetric::new())
            .metric_numeric(LossMetric::new())
            .summary()
            .build(self.clone());

        eval.eval("Evaluation", dataloader);

        // TODO: this doesn't give a summary?
    }
}
