use serde::Deserialize;
use std::time::Duration;
use yahoo_finance_api::time::{Date, Month};
use yahoo_finance_api::{YahooConnector, YahooConnectorBuilder};

pub fn parse_date(input: &str) -> Date {
    let days = &input[0..2];
    let month = &input[3..5];
    let year = &input[6..10];

    Date::from_calendar_date(
        year.parse().expect("Failed to parse year"),
        Month::try_from(
            month
                .parse::<u8>()
                .expect("Failed to parse month into integer"),
        )
        .expect("Failed to parse month"),
        days.parse().expect("Failed to parse day"),
    )
    .expect("Failed to parse date")
}

pub fn split_tags<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    Ok(s.split(';').map(|item| item.to_string()).collect())
}

pub fn interval_to_duration(interval: &str) -> Option<Duration> {
    match interval {
        "1d" => Some(Duration::from_hours(24)),
        _ => None,
    }
}

pub fn yahoo(timeout: u64) -> YahooConnector {
    log::info!("Connecting to the Yahoo Finance API...");

    YahooConnectorBuilder::new()
        .timeout(Duration::from_millis(timeout))
        .build()
        .expect("Failed to build Yahoo connector")
}
