use crate::utils::{DATE_FORMAT, parse_date, round_to, yahoo};
use csv::Trim;
use mirada_lib::data::{DataKey, StockData};
use mirada_lib::database::Database;
use serde::{Deserialize, Serialize};
use yahoo_finance_api::YahooConnector;
use yahoo_finance_api::time::OffsetDateTime;

pub fn fetch(
    database: String,
    timeout: u64,
    dont_skip: bool,
    start: Option<String>,
    end: Option<String>,
    ticker: Option<String>,
    file: Option<String>,
) {
    let yahoo = yahoo(timeout);

    let database = Database::new(database);

    if let Some(file) = file {
        log::info!("Reading input arguments from '{file}'...");

        let records = csv::ReaderBuilder::new()
            .trim(Trim::All)
            .buffer_capacity(1024)
            .from_path(file)
            .expect("Failed to read CSV file")
            .deserialize()
            .map(|rec| rec.expect("Failed to deserialize record"))
            .collect::<Vec<Record>>();

        let length = records.len() as f32;

        for (idx, record) in records.into_iter().enumerate() {
            let start = parse_date(&record.start).midnight().assume_utc();
            let end = parse_date(&record.end).midnight().assume_utc();

            let key = DataKey::new(record.ticker.clone(), start, end);

            let progress = round_to((idx as f32 / length) * 100.0, 2);
            log::info!("Progress: {progress}%");

            if !dont_skip && database.get(key.clone()).is_some() {
                log::warn!("Key {} already exists. Skipping...", key);
                continue;
            } else {
                let data = fetch_data(&yahoo, start, end, record.ticker.clone(), true);
                database.insert(key, data);
            }
        }
    } else {
        let start = start.expect("'start' argument must be provided");
        let end = end.expect("'end' argument must be provided");
        let ticker = ticker.expect("'ticker' argument must be provided");

        let start = parse_date(&start).midnight().assume_utc();
        let end = parse_date(&end).midnight().assume_utc();

        let data = fetch_data(&yahoo, start, end, ticker.clone(), true);
        database.insert(DataKey::new(ticker, start, end), data);
    }
}

pub fn fetch_data(
    yahoo: &YahooConnector,
    start: OffsetDateTime,
    end: OffsetDateTime,
    ticker: String,
    training: bool,
) -> StockData {
    log::info!(
        "Sending request for '{ticker}' from {} to {}...",
        start.format(DATE_FORMAT).expect("Failed to format start"),
        end.format(DATE_FORMAT).expect("Failed to format end")
    );
    let response = yahoo
        .get_quote_history(&ticker, start, end)
        .expect("Failed to get quote history");

    let quotes = response.quotes().expect("Failed to get response result");

    assert!(!quotes.is_empty(), "Got empty response for '{ticker}'");

    log::info!("Collecting request data...");
    let mut opens = Vec::with_capacity(quotes.len());
    let mut closes = Vec::with_capacity(quotes.len());
    let mut volumes = Vec::with_capacity(quotes.len());
    let mut highs = Vec::with_capacity(quotes.len());
    let mut lows = Vec::with_capacity(quotes.len());

    for q in quotes {
        opens.push(q.open as f32);
        closes.push(q.adjclose as f32);
        volumes.push(q.volume as f32);
        highs.push(q.high as f32);
        lows.push(q.low as f32);

        assert!(
            q.open != 0.0
                && q.adjclose != 0.0
                && q.volume != 0
                && q.high != 0.0
                && q.low != 0.0
                && !q.open.is_nan()
                && !q.adjclose.is_nan()
                && !q.high.is_nan()
                && !q.low.is_nan(),
            "Got invalid response for '{ticker}'"
        );
    }

    StockData::new(opens, closes, volumes, highs, lows, training)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Record {
    pub ticker: String,
    pub start: String,
    pub end: String,
}
