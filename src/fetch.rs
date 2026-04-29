use crate::csv::Record;
use crate::utils::{DATE_FORMAT, parse_date, yahoo};
use csv::Trim;
use mirada_lib::consts::OTHER_STOCKS;
use mirada_lib::data::{DataKey, StockData};
use mirada_lib::database::Database;
use rayon::prelude::*;
use std::sync::atomic::AtomicU64;
use std::time::Duration;
use yahoo_finance_api::YahooConnector;
use yahoo_finance_api::time::OffsetDateTime;

pub fn fetch(
    database: String,
    timeout: u64,
    serial: bool,
    retry: u8,
    _override: bool,
    file: String,
) {
    let yahoo = yahoo(timeout);

    let database = Database::new(database);

    log::info!("Reading input arguments from '{file}'...");

    let records = csv::ReaderBuilder::new()
        .trim(Trim::All)
        .buffer_capacity(1024)
        .from_path(file)
        .expect("Failed to read CSV file")
        .deserialize()
        .map(|rec| rec.expect("Failed to deserialize record"))
        .collect::<Vec<Record>>();

    // Multiply by 1_000_000 to effectively create an atomic float
    let progress_per_item = (100.0 / records.len() as f32 * 1_000_000.0) as u64;
    let progress = AtomicU64::new(0);

    let process = |record: Record| {
        let start = parse_date(&record.start).midnight().assume_utc();
        let end = parse_date(&record.end).midnight().assume_utc();

        let key = DataKey::new(record.ticker.clone(), start, end);

        let progress = (progress.fetch_add(progress_per_item, std::sync::atomic::Ordering::SeqCst)
            + progress_per_item) as f32;
        let percentage = (progress / 10_000.0).round() / 100.0;

        log::info!("Progress: {}%", percentage);

        if !_override && database.get(key.clone()).is_some() {
            log::warn!("Key {} already exists. Skipping...", key);
            None
        } else {
            let target = fetch_stock(&yahoo, start, end, record.ticker, true, retry);
            let others = record
                .others
                .into_iter()
                .map(|other| fetch_stock(&yahoo, start, end, other, true, retry))
                .collect::<Vec<_>>();

            Some((key, target, others))
        }
    };

    let items = if serial {
        records.into_iter().filter_map(process).collect::<Vec<_>>()
    } else {
        records
            .into_par_iter()
            .filter_map(process)
            .collect::<Vec<_>>()
    };

    let items = items.into_iter().map(|(key, target, others)| {
        let other_closes = others.into_iter().map(|res| res.closes).collect::<Vec<_>>();

        assert_eq!(
            other_closes.len(),
            OTHER_STOCKS,
            "Must provide exactly {OTHER_STOCKS} stocks, but {} were given",
            other_closes.len()
        );

        (
            key,
            StockData::new(
                target.opens,
                target.closes,
                other_closes.try_into().unwrap(),
                target.volumes,
                target.highs,
                target.lows,
                target.date_range,
                target.training,
            ),
        )
    });

    for (key, data) in items {
        database.insert(key, data);
    }
}

pub fn fetch_stock(
    yahoo: &YahooConnector,
    start: OffsetDateTime,
    end: OffsetDateTime,
    ticker: String,
    training: bool,
    mut retry: u8,
) -> FetchResult {
    log::info!(
        "Sending request for '{ticker}' from {} to {}...",
        start.format(DATE_FORMAT).expect("Failed to format start"),
        end.format(DATE_FORMAT).expect("Failed to format end")
    );

    let mut response = yahoo.get_quote_history(&ticker, start, end);

    while retry > 0 && response.is_err() {
        log::warn!("Response for {ticker} failed! Retrying ({retry} retries left)...");

        std::thread::sleep(Duration::from_millis(100));
        retry -= 1;
        response = yahoo.get_quote_history(&ticker, start, end);
    }

    let response = response.expect("Failed to fetch data from Yahoo Finance API");

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
            !q.open.is_nan() && !q.adjclose.is_nan() && !q.high.is_nan() && !q.low.is_nan(),
            "Got invalid response for '{ticker}'"
        );
    }

    FetchResult {
        opens,
        closes,
        volumes,
        highs,
        lows,
        date_range: (start, end),
        training,
    }
}

#[derive(Clone, Debug)]
pub struct FetchResult {
    pub opens: Vec<f32>,
    pub closes: Vec<f32>,
    pub volumes: Vec<f32>,
    pub highs: Vec<f32>,
    pub lows: Vec<f32>,
    pub date_range: (OffsetDateTime, OffsetDateTime),
    pub training: bool,
}
