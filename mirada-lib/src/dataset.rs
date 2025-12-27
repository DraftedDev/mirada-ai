use crate::data::{DataKey, OTHER_STOCKS, StockData};
use crate::database::Database;
use burn::data::dataset::Dataset;
use burn::prelude::Backend;
use time::{Date, OffsetDateTime};

#[derive(Default, Clone)]
pub struct StockDataset {
    items: Vec<DataItem>,
}

impl StockDataset {
    pub fn add_item(&mut self, ticker: String, start: Date, end: Date, others: Vec<String>) {
        assert_eq!(
            others.len(),
            OTHER_STOCKS,
            "Can only process {OTHER_STOCKS} other stocks"
        );

        self.items.push(DataItem::new(ticker, start, end, others));
    }
}

impl Dataset<DataItem> for StockDataset {
    fn get(&self, index: usize) -> Option<DataItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[derive(Debug, Clone)]
pub struct DataItem {
    ticker: String,
    start: OffsetDateTime,
    end: OffsetDateTime,
    others: Vec<String>,
}

impl DataItem {
    pub fn new(ticker: String, start: Date, end: Date, others: Vec<String>) -> Self {
        Self {
            ticker,
            start: start.midnight().assume_utc(),
            end: end.midnight().assume_utc(),
            others,
        }
    }

    pub fn to_stock_data<B: Backend>(
        self,
        database: &Database,
        device: &B::Device,
        interval: String,
    ) -> Option<StockData> {
        let stock = database.get(DataKey::new(
            self.ticker,
            self.start,
            self.end,
            interval.clone(),
        ))?;

        let others = self
            .others
            .iter()
            .map(|other| {
                let key = DataKey::new(other.clone(), self.start, self.end, interval.clone());
                database.get(key).expect("Failed to get other data")
            })
            .collect::<Vec<_>>();

        Some(stock.merge::<B>(others, device))
    }
}
