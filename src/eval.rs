use crate::train::build_dataset;
use csv::{ReaderBuilder, Trim};
use mirada_lib::AutodiffBackend;
use mirada_lib::database::Database;
use mirada_lib::eval::EvalConfig;
use mirada_lib::model::{Model, ModelConfig};
use std::path::Path;

pub fn eval(
    database: String,
    dataset: String,
    model_cfg: String,
    eval_cfg: String,
    artifacts: String,
) {
    let database = Database::new(database);

    log::info!("Reading evaluation CSV file...");
    let file = ReaderBuilder::new()
        .buffer_capacity(512)
        .trim(Trim::All)
        .from_path(dataset)
        .expect("Failed to read CSV file.");
    log::info!("Creating training dataset...");
    let dataset = build_dataset(file);

    log::info!("Loading configuration files...");
    let model_cfg = ModelConfig::load_or_create(model_cfg);
    let training_cfg = EvalConfig::load_or_create(eval_cfg);

    log::info!("Creating device...");
    let device = mirada_lib::Device::default();

    let model: Model<AutodiffBackend> =
        Model::load(model_cfg, Path::new(&artifacts).join("model"), &device);

    log::info!("Evaluating model...");
    model.eval(training_cfg, database, dataset, artifacts, device);
}
