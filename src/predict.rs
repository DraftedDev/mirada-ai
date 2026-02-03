use crate::fetch::fetch_data;
use crate::utils::{DATE_FORMAT, parse_date, yahoo};
use mirada_lib::consts::HORIZON;
use mirada_lib::model::{Model, ModelConfig};
use mirada_lib::{Backend, Device};
use std::path::Path;
use time::Duration;

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
        let data = fetch_data(&yahoo, start, end, ticker.clone(), false);

        let others = others
            .into_iter()
            .map(|other| fetch_data(&yahoo, start, end, other, false))
            .collect::<Vec<_>>();

        data.merge::<Backend>(others, &device)
    };

    log::info!("Predicting price for '{ticker}'....");
    let is_up = model.infer_up(data, &device);

    let approx_date = end + Duration::days(1) * HORIZON as u32;

    log::info!(
        "### MODEL PREDICTION ###\n\
    \tStock: {}\n\
    \tMove: {}\n\
    \tLast Observed: {}\n\
    \tHorizon: {} bars\n\
    \tApprox. Date: {}",
        ticker,
        if is_up { "Up (+)" } else { "Down (-)" },
        end.format(DATE_FORMAT).expect("Failed to format end date"),
        HORIZON,
        approx_date
            .format(DATE_FORMAT)
            .expect("Failed to format approx date")
    );
}
