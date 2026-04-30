use crate::fetch::fetch_stock;
use crate::utils;
use crate::utils::{CHRONO_DATE_FORMAT, DATE_FORMAT, parse_date, yahoo};
use mirada_lib::consts::{HORIZON, OTHER_STOCKS};
use mirada_lib::data::StockData;
use mirada_lib::model::{Model, ModelConfig};
use mirada_lib::{Backend, Device};
use std::path::Path;
use trading_calendar::{Market, TradingCalendar};

pub fn predict(
    timeout: u64,
    artifacts: String,
    model_cfg: String,
    start: String,
    end: String,
    ticker: String,
    others: Vec<String>,
) {
    let yahoo = yahoo(timeout);

    let start = parse_date(&start).midnight().assume_utc();
    let end = parse_date(&end).midnight().assume_utc();

    log::info!("Creating device...");
    let device = Device::default();

    log::info!("Loading Model Configuration...");
    let model_cfg = ModelConfig::load_or_create(model_cfg);

    let model_path = Path::new(&artifacts).join("model");

    log::info!("Loading Model from {}...", model_path.display());
    let model = Model::<Backend>::load(model_cfg, model_path, &device);

    let data = {
        let data = fetch_stock(&yahoo, start, end, ticker.clone(), false, 0);

        let others: [Vec<f32>; OTHER_STOCKS] = others
            .into_iter()
            .map(|other| fetch_stock(&yahoo, start, end, other, false, 0).closes)
            .collect::<Vec<_>>()
            .try_into()
            .expect("Must provide OTHER_STOCKS other stocks");

        StockData::new(
            data.opens,
            data.closes,
            others,
            data.volumes,
            data.highs,
            data.lows,
            data.date_range,
            false,
        )
    };

    log::info!("Predicting price for '{ticker}'....");
    let result = model.infer(data, &device);

    let class = match result {
        0 => "Down (-)",
        1 => "Neutral",
        2 => "Up (+)",
        _ => unreachable!(),
    };

    let calendar = TradingCalendar::new(Market::NASDAQ).expect("Failed to get NASDAQ calendar");
    let mut approx = utils::date_to_naive(end.date());

    for _ in 0..HORIZON {
        approx = calendar.next_trading_day(approx);
    }

    log::info!(
        "### MODEL PREDICTION ###\n\
    \tStock: {}\n\
    \tMove: {}\n\
    \tLast Observed: {}\n\
    \tHorizon: {} bars\n\
    \tApprox. Date: {}",
        ticker,
        class,
        end.format(DATE_FORMAT).expect("Failed to format end date"),
        HORIZON,
        approx.format(CHRONO_DATE_FORMAT).to_string()
    );
}
