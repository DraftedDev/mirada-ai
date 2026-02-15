use crate::utils::parse_date;
use csv::{Reader, ReaderBuilder, Trim};
use mirada_lib::AutodiffBackend;
use mirada_lib::database::Database;
use mirada_lib::dataset::StockDataset;
use mirada_lib::model::{Model, ModelConfig};
use mirada_lib::training::TrainingConfig;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;

pub fn train(
    database: String,
    train_dataset: String,
    valid_dataset: String,
    model_cfg: String,
    training_cfg: String,
    artifacts: String,
    cleanup: bool,
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

    let model: Model<AutodiffBackend> = if cleanup {
        log::info!("Cleaning up artifact directory...");
        std::fs::remove_dir_all(&artifacts).expect("Failed to cleanup artifacts directory");
        std::fs::create_dir_all(&artifacts).expect("Failed to create artifacts directory");

        model_cfg.init(&device)
    } else {
        Model::load(model_cfg, Path::new(&artifacts).join("model"), &device)
    };

    log::info!("Training model...");
    model.train(
        training_cfg,
        database,
        train_dataset,
        valid_dataset,
        artifacts,
        device,
    );
}

pub fn build_dataset(mut reader: Reader<File>) -> StockDataset {
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
