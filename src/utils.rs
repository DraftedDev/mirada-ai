use serde::Deserialize;
use std::fmt::Debug;
use std::str::FromStr;
use std::time::Duration;
use time::format_description::BorrowedFormatItem;
use yahoo_finance_api::time::Date;
use yahoo_finance_api::{YahooConnector, YahooConnectorBuilder};

pub const DATE_FORMAT: &[BorrowedFormatItem] =
    time::macros::format_description!("[day].[month].[year]");

pub fn parse_date(input: &str) -> Date {
    Date::parse(input, DATE_FORMAT).expect("Failed to parse date")
}

pub fn split_tags<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    Ok(s.split(';').map(|item| item.to_string()).collect())
}

pub fn join_tags<S>(tags: &[String], serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let s = tags.join(";");
    serializer.serialize_str(&s)
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

pub fn env_or_default<T>(var: &str, default: T) -> T
where
    T: FromStr,
    T::Err: Debug,
{
    std::env::var(var)
        .map(|s| T::from_str(&s).unwrap_or_else(|_| panic!("Failed to parse env var {var}")))
        .unwrap_or(default)
}

pub fn round_to(value: f32, precision: i32) -> f32 {
    let factor = 10.0_f32.powi(precision);

    (value * factor).round() / factor
}
