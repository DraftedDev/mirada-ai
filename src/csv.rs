use crate::utils::{DATE_FORMAT, parse_date};
use crate::{fetch, train};
use csv::{ReaderBuilder, WriterBuilder};
use serde::Serialize;
use std::collections::BTreeMap;
use std::time::Duration;

pub fn fetch(
    output: String,
    start: String,
    end: String,
    length: u64,
    shift: u64,
    tickers: Vec<String>,
) {
    let start = parse_date(&start);
    let end = parse_date(&end);

    let window = Duration::from_hours(24 * length);
    let step = Duration::from_hours(24 * shift);

    let mut rows = Vec::with_capacity(tickers.len());
    let mut current_start = start;

    log::info!(
        "Generating rows (window = {} days, shift = {} days)...",
        length,
        shift
    );

    while current_start < end {
        let current_end = current_start + window;

        if current_end > end {
            log::warn!(
                "Item from {current_start} to {current_end} ignored, due to insufficient data."
            );
            break;
        }

        for ticker in &tickers {
            rows.push(fetch::Record {
                ticker: ticker.clone(),
                start: current_start
                    .format(DATE_FORMAT)
                    .expect("Failed to format start date"),
                end: current_end
                    .format(DATE_FORMAT)
                    .expect("Failed to format end date"),
            });
        }

        // advance by shift and not by window length
        current_start += step;
    }

    log::info!("Writing {} rows to '{output}'...", rows.len());
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
        "Successfully split {} rows into {}/{}",
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
