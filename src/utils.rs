use serde::Deserialize;
use std::fmt::Debug;
use std::str::FromStr;
use std::time::Duration;
use time::format_description::BorrowedFormatItem;
use trading_calendar::NaiveDate;
use yahoo_finance_api::time::Date;
use yahoo_finance_api::{YahooConnector, YahooConnectorBuilder};

pub const CHRONO_DATE_FORMAT: &str = "%d.%m.%Y";

pub const DATE_FORMAT: &[BorrowedFormatItem] =
    time::macros::format_description!("[day].[month].[year]");

pub fn naive_to_date(naive: NaiveDate) -> Date {
    parse_date(&naive.format(CHRONO_DATE_FORMAT).to_string())
}

pub fn date_to_naive(date: Date) -> NaiveDate {
    NaiveDate::parse_from_str(
        date.format(DATE_FORMAT).unwrap().as_str(),
        CHRONO_DATE_FORMAT,
    )
    .unwrap()
}

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

pub fn yahoo(timeout: u64) -> YahooConnector {
    log::info!("Connecting to the Yahoo Finance API...");

    YahooConnectorBuilder::new()
        .timeout(Duration::from_millis(timeout))
        .user_agent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:149.0) Gecko/20100101 Firefox/149.0",
        )
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
