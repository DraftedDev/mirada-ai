use crate::fetch::fetch_data;
use crate::utils::{interval_to_duration, parse_date, yahoo};
use mirada_lib::data::HORIZON;
use mirada_lib::model::{Model, ModelConfig};
use mirada_lib::{Backend, Device};
use std::path::Path;

pub fn predict(
    timeout: u64,
    artifacts: String,
    model_cfg: String,
    interval: String,
    start: String,
    end: String,
    ticker: String,
    others: Vec<String>,
) {
    let yahoo = yahoo(timeout);

    let start = parse_date(&start).midnight().assume_utc();
    let end = parse_date(&end).midnight().assume_utc();

    let interval_dur = interval_to_duration(&interval).expect("Invalid interval");

    log::info!("Creating device...");
    let device = Device::default();

    log::info!("Loading Model Configuration...");
    let model_cfg = ModelConfig::load_or_create(model_cfg);

    let model_path = Path::new(&artifacts).join("model");

    log::info!("Loading Model from {}...", model_path.display());
    let model = Model::<Backend>::load(model_cfg, model_path, &device);

    let data = {
        let data = fetch_data(&yahoo, start, end, interval.clone(), ticker.clone());

        let others = others
            .into_iter()
            .map(|other| fetch_data(&yahoo, start, end, interval.clone(), other))
            .collect::<Vec<_>>();

        data.merge::<Backend>(others, &device)
    };

    log::info!("Predicting price for '{ticker}'....");
    let result = model.infer_price(data, &device);

    let pred_date = end + interval_dur * HORIZON as u32;

    log::info!(
        "### MODEL PREDICTION ###\n\
\tStock: {ticker}\n\
\tPrice: {result}\n\
\tDate: {pred_date}"
    );
}
