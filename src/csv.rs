use crate::utils::{CHRONO_DATE_FORMAT, parse_date};
use crate::{fetch, train};
use csv::{ReaderBuilder, WriterBuilder};
use serde::Serialize;
use std::collections::BTreeMap;
use trading_calendar::{Market, NaiveDate, TradingCalendar};

pub fn fetch(
    output: String,
    start: String,
    end: String,
    length: u64,
    shift: u64,
    tickers: Vec<String>,
) {
    let (start_y, start_o) = parse_date(&start).to_ordinal_date();
    let start =
        NaiveDate::from_yo_opt(start_y, start_o as u32).expect("Failed to parse start date");
    let (end_y, end_o) = parse_date(&end).to_ordinal_date();
    let end = NaiveDate::from_yo_opt(end_y, end_o as u32).expect("Failed to parse end date");

    // initialize NASDAQ trading calendar
    let calendar = TradingCalendar::new(Market::NASDAQ).expect("Failed to get NASDAQ calendar");

    log::info!(
        "Generating rows (window = {} trading days, shift = {} trading days)...",
        length,
        shift
    );

    let mut rows = Vec::new();
    let mut current_start = calendar.next_trading_day(start);

    while current_start < end {
        // get the end date of the window (length trading days after current_start)
        let mut current_end = current_start;

        for _ in 0..(length - 1) {
            current_end = calendar.next_trading_day(current_end);
        }

        if current_end >= end {
            log::warn!(
                "Window from {} to {} exceeds end date. Skipping...",
                current_start,
                current_end
            );
            break;
        }

        // add a row for each ticker
        for ticker in &tickers {
            rows.push(fetch::Record {
                ticker: ticker.clone(),
                start: current_start.format(CHRONO_DATE_FORMAT).to_string(),
                end: current_end.format(CHRONO_DATE_FORMAT).to_string(),
            });
        }

        // advance start by `shift` trading days
        for _ in 0..shift {
            current_start = calendar.next_trading_day(current_start);
        }
    }

    log::info!("Writing {} rows to '{}'", rows.len(), output);
    write_csv_to(output, &rows);
}

pub fn train(out1: String, out2: String, percent: f32, input: String) {
    let mut groups: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();

    log::info!("Reading input file '{input}'...");
    let mut rdr = ReaderBuilder::new()
        .from_path(&input)
        .expect("Failed to open input file");

    for result in rdr.deserialize() {
        let record: fetch::Record = result.expect("Error parsing record");
        groups
            .entry((record.start, record.end))
            .or_default()
            .push(record.ticker);
    }

    log::info!("Processing input rows...");
    let mut processed_rows: Vec<train::Record> = groups
        .into_iter()
        .filter_map(|((start, end), tickers)| {
            if tickers.is_empty() {
                None
            } else {
                let primary = tickers[0].clone();
                let others = tickers[1..].to_vec();
                Some(train::Record {
                    ticker: primary,
                    start,
                    end,
                    others,
                })
            }
        })
        .collect();

    let total_groups = processed_rows.len();
    let split_idx = (total_groups as f32 * (percent / 100.0)) as usize;

    let rows2 = processed_rows.split_off(split_idx);
    let rows1 = processed_rows;

    write_csv_to(out1, &rows1);
    write_csv_to(out2, &rows2);

    log::info!(
        "Successfully split {} full rows into {}/{}",
        total_groups,
        rows1.len(),
        rows2.len()
    );
}

fn write_csv_to<S: Serialize>(out: String, rows: &[S]) {
    if out.as_str() == "stdout" {
        println!("########## CSV DATA ##########");
        let mut writer = WriterBuilder::new()
            .buffer_capacity(rows.len() * 512)
            .from_writer(std::io::stdout());

        for row in rows {
            writer.serialize(row).expect("Failed to serialize row");
        }

        writer.flush().expect("Failed to flush to stdout");
    } else {
        let mut writer = WriterBuilder::new()
            .buffer_capacity(rows.len() * 512)
            .from_path(&out)
            .expect("Failed to create writer");

        for row in rows {
            writer.serialize(row).expect("Failed to serialize row");
        }

        writer.flush().expect("Failed to flush to file");
    }
}
