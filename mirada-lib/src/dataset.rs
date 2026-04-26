use crate::data::{DataKey, StockData};
use crate::database::Database;
use burn::data::dataset::Dataset;
use burn::prelude::Backend;
use time::{Date, OffsetDateTime};

#[derive(Default, Clone)]
pub struct StockDataset {
    items: Vec<DataItem>,
}

impl StockDataset {
    pub fn add_item(&mut self, ticker: String, start: Date, end: Date) {
        self.items.push(DataItem::new(ticker, start, end));
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
}

impl DataItem {
    pub fn new(ticker: String, start: Date, end: Date) -> Self {
        Self {
            ticker,
            start: start.midnight().assume_utc(),
            end: end.midnight().assume_utc(),
        }
    }

    pub fn to_stock_data<B: Backend>(self, database: &Database) -> Option<StockData> {
        let stock = database.get(DataKey::new(self.ticker, self.start, self.end))?;

        Some(stock)
    }
}
