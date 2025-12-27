use crate::utils::parse_date;
use csv::{Reader, ReaderBuilder, Trim};
use mirada_lib::AutodiffBackend;
use mirada_lib::database::Database;
use mirada_lib::dataset::StockDataset;
use mirada_lib::model::{Model, ModelConfig};
use mirada_lib::training::TrainingConfig;
use serde::{Deserialize, Serialize};
use std::fs::File;

pub fn train(
    database: String,
    train_dataset: String,
    valid_dataset: String,
    model_cfg: String,
    training_cfg: String,
    interval: String,
    artifacts: String,
) {
    let database = Database::new(database);

    log::info!("Reading training CSV file...");
    let train_file = ReaderBuilder::new()
        .buffer_capacity(512)
        .trim(Trim::All)
        .from_path(train_dataset)
        .expect("Failed to read CSV file.");
    log::info!("Creating training dataset...");
    let train_dataset = build_dataset(train_file);

    log::info!("Reading validation CSV file...");
    let valid_file = ReaderBuilder::new()
        .buffer_capacity(512)
        .trim(Trim::All)
        .from_path(valid_dataset)
        .expect("Failed to read CSV file.");
    log::info!("Creating training dataset...");
    let valid_dataset = build_dataset(valid_file);

    log::info!("Loading configuration files...");
    let model_cfg = ModelConfig::load_or_create(model_cfg);
    let training_cfg = TrainingConfig::load_or_create(training_cfg);

    log::info!("Creating device...");
    let device = mirada_lib::Device::default();

    log::info!("Initializing model...");
    let model: Model<AutodiffBackend> = model_cfg.init(&device);

    log::info!("Training model...");
    model.train(
        training_cfg,
        database,
        train_dataset,
        valid_dataset,
        interval,
        artifacts,
        device,
    );
}

fn build_dataset(mut reader: Reader<File>) -> StockDataset {
    let mut dataset = StockDataset::default();

    for record in reader.deserialize::<Record>() {
        let record = record.expect("Failed to read CSV record.");

        let start = parse_date(&record.start);
        let end = parse_date(&record.end);

        dataset.add_item(record.ticker, start, end, record.others);
    }

    dataset
}

#[derive(Serialize, Deserialize)]
struct Record {
    ticker: String,
    start: String,
    end: String,
    #[serde(deserialize_with = "crate::utils::split_tags")]
    others: Vec<String>,
}
