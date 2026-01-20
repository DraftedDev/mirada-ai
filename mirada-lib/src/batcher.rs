use crate::database::Database;
use crate::dataset::DataItem;
use burn::Tensor;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::Backend;
use burn::tensor::Int;

#[derive(Clone)]
pub struct StockBatcher {
    database: Database,
}

impl StockBatcher {
    pub fn new(database: Database) -> Self {
        Self { database }
    }
}

impl<B: Backend> Batcher<B, DataItem, DataBatch<B>> for StockBatcher {
    fn batch(&self, items: Vec<DataItem>, device: &B::Device) -> DataBatch<B> {
        assert!(!items.is_empty(), "No items to batch");

        let mut features = Vec::with_capacity(items.len());
        let mut targets = Vec::with_capacity(items.len());

        for item in items {
            let item = item
                .to_stock_data::<B>(&self.database, device)
                .expect("Failed to get stock data");

            let (feature, target) = item.into_tensors(device);
            features.push(feature);
            targets.push(target);
        }

        DataBatch {
            features: Tensor::cat(features, 0),
            targets: Tensor::cat(targets, 0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DataBatch<B: Backend> {
    pub features: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}
