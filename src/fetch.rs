use crate::utils::{DATE_FORMAT, parse_date, yahoo};
use csv::Trim;
use mirada_lib::data::{DataKey, StockData};
use mirada_lib::database::Database;
use serde::{Deserialize, Serialize};
use yahoo_finance_api::YahooConnector;
use yahoo_finance_api::time::OffsetDateTime;

pub fn fetch(
    database: String,
    timeout: u64,
    interval: String,
    start: Option<String>,
    end: Option<String>,
    ticker: Option<String>,
    file: Option<String>,
) {
    let yahoo = yahoo(timeout);

    let database = Database::new(database);

    if let Some(file) = file {
        log::info!("Reading input arguments from '{file}'...");

        let mut reader = csv::ReaderBuilder::new()
            .trim(Trim::All)
            .buffer_capacity(1024)
            .from_path(file)
            .expect("Failed to read CSV file");

        for record in reader.deserialize::<Record>() {
            let record =
                record.expect("Failed to deserialize record with format `ticker,start,end`");

            let start = parse_date(&record.start).midnight().assume_utc();
            let end = parse_date(&record.end).midnight().assume_utc();

            let data = fetch_data(&yahoo, start, end, interval.clone(), record.ticker.clone());

            database.insert(
                DataKey::new(record.ticker, start, end, interval.clone()),
                data,
            );
        }
    } else {
        let start = start.expect("'start' argument must be provided");
        let end = end.expect("'end' argument must be provided");
        let ticker = ticker.expect("'ticker' argument must be provided");

        let start = parse_date(&start).midnight().assume_utc();
        let end = parse_date(&end).midnight().assume_utc();

        let data = fetch_data(&yahoo, start, end, interval.clone(), ticker.clone());
        database.insert(DataKey::new(ticker, start, end, interval), data);
    }
}

pub fn fetch_data(
    yahoo: &YahooConnector,
    start: OffsetDateTime,
    end: OffsetDateTime,
    interval: String,
    ticker: String,
) -> StockData {
    log::info!(
        "Sending request for '{ticker}' from {} to {} with interval {interval}...",
        start.format(DATE_FORMAT).expect("Failed to format start"),
        end.format(DATE_FORMAT).expect("Failed to format end")
    );
    let response = yahoo
        .get_quote_history_interval(&ticker, start, end, &interval)
        .expect("Failed to get quote history");

    let quotes = response.quotes().expect("Failed to get response result");

    log::info!("Collecting request data...");
    let mut opens = Vec::with_capacity(quotes.len());
    let mut closes = Vec::with_capacity(quotes.len());
    let mut volumes = Vec::with_capacity(quotes.len());
    let mut highs = Vec::with_capacity(quotes.len());
    let mut lows = Vec::with_capacity(quotes.len());

    for q in quotes {
        opens.push(q.open as f32);
        closes.push(q.close as f32);
        volumes.push(q.volume as f32);
        highs.push(q.high as f32);
        lows.push(q.low as f32);
    }

    StockData::new(opens, closes, volumes, highs, lows)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Record {
    pub ticker: String,
    pub start: String,
    pub end: String,
}
