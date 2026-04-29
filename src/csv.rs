use crate::utils::{CHRONO_DATE_FORMAT, parse_date};
use csv::WriterBuilder;
use rand::RngExt;
use rand::prelude::SliceRandom;
use serde::{Deserialize, Serialize};
use trading_calendar::{Market, NaiveDate, TradingCalendar};

pub fn csv(
    output: String,
    start: String,
    end: String,
    length: u64,
    samples: usize,
    ticker: String,
    others: Vec<String>,
) {
    let (start_y, start_o) = parse_date(&start).to_ordinal_date();
    let start =
        NaiveDate::from_yo_opt(start_y, start_o as u32).expect("Failed to parse start date");

    let (end_y, end_o) = parse_date(&end).to_ordinal_date();
    let end = NaiveDate::from_yo_opt(end_y, end_o as u32).expect("Failed to parse end date");

    let calendar = TradingCalendar::new(Market::NASDAQ).expect("Failed to get NASDAQ calendar");

    let mut rng = rand::rng();

    log::info!(
        "Generating rows (window = {}, samples = {})...",
        length,
        samples,
    );

    let mut trading_days = Vec::new();
    let mut d = start;

    while d <= end {
        trading_days.push(d);
        d = calendar.next_trading_day(d);
    }

    let max_start = trading_days.len().saturating_sub(length as usize);

    if max_start == 0 {
        panic!("Not enough data to build any window");
    }

    let mut indices: Vec<usize> = (0..max_start).collect();
    indices.shuffle(&mut rng);

    let mut rows = Vec::with_capacity(samples);

    for &start_idx in indices.iter().take(samples.min(max_start)) {
        let current_start = trading_days[start_idx];
        let current_end = trading_days[start_idx + length as usize - 1];

        rows.push(Record {
            ticker: ticker.clone(),
            start: current_start.format(CHRONO_DATE_FORMAT).to_string(),
            end: current_end.format(CHRONO_DATE_FORMAT).to_string(),
            others: others.clone(),
        });
    }

    if samples > max_start {
        log::info!("Generating duplicates to match samples...");

        for _ in 0..(samples - max_start) {
            let start_idx = rng.random_range(0..max_start);

            let current_start = trading_days[start_idx];
            let current_end = trading_days[start_idx + length as usize - 1];

            rows.push(Record {
                ticker: ticker.clone(),
                start: current_start.format(CHRONO_DATE_FORMAT).to_string(),
                end: current_end.format(CHRONO_DATE_FORMAT).to_string(),
                others: others.clone(),
            });
        }
    }

    log::info!("Writing {} rows to '{}'", rows.len(), output);
    write_csv_to(output, &rows);
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Record {
    pub ticker: String,
    pub start: String,
    pub end: String,
    #[serde(
        deserialize_with = "crate::utils::split_tags",
        serialize_with = "crate::utils::join_tags"
    )]
    pub others: Vec<String>,
}
