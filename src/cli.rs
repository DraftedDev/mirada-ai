use clap::{Args, Parser, Subcommand};
use mirada_lib::database;

/// Interface to the Miranda AI.
#[derive(Parser)]
#[command(
    name = "mirada-ai",
    version = env!("CARGO_PKG_VERSION"),
    author = "Mikail Plotzky <mp@ypon.com>"
)]
pub struct MirandaCli {
    /// The command to run.
    #[command(subcommand)]
    pub command: Command,
    /// The path to the database to use for various operations.
    #[arg(short = 'd', long = "database", default_value = database::DATABASE_PATH)]
    pub database: String,
    /// The timeout to use for web requests.
    #[arg(short = 't', long = "timeout", default_value = "5000")]
    pub timeout: u64,
}

/// The commands available.
#[derive(Subcommand)]
pub enum Command {
    /// Fetches data from the Yahoo Finance API.
    Fetch(FetchArgs),
    /// Trains a model on the given dataset.
    Train(TrainArgs),
    /// Makes a prediction for the given stock data.
    Predict(PredictArgs),
    /// Dumps the given data kind out to the console.
    Dump(DumpArgs),
}

/// Arguments for the fetch command.
#[derive(Args)]
pub struct FetchArgs {
    /// The interval for the request.
    #[arg(short = 'i', long = "interval", default_value = "1d")]
    pub interval: String,
    /// Whether to fetch multiple quotes using the given CSV file content as input args.
    #[arg(
        short = 'f',
        long = "file",
        conflicts_with_all = ["start", "end", "ticker"],
        required_unless_present_all = ["start", "end", "ticker"]
    )]
    pub file: Option<String>,
    /// The start date to fetch the data from in `dd-MM-YYYY` format.
    #[arg(required_unless_present = "file")]
    pub start: Option<String>,
    /// The end date to fetch the data from in `dd-MM-YYYY` format.
    #[arg(required_unless_present = "file")]
    pub end: Option<String>,
    /// The ticker to fetch the data for.
    #[arg(required_unless_present = "file")]
    pub ticker: Option<String>,
}

/// Arguments for the train command.
#[derive(Args)]
pub struct TrainArgs {
    /// Path to the model configuration. Will be created if it doesn't exist yet.
    #[arg(short = 'm', long = "model", default_value = "./config/model.json")]
    pub model: String,
    /// Path to the training configuration. Will be created if it doesn't exist yet.
    #[arg(
        short = 't',
        long = "training",
        default_value = "./config/training.json"
    )]
    pub training: String,
    /// The interval for the request.
    #[arg(short = 'i', long = "interval", default_value = "1d")]
    pub interval: String,
    /// The artifacts directory to use for training.
    #[arg(short = 'a', long = "artifacts", default_value = "./artifacts")]
    pub artifacts: String,
    /// Whether to cleanup the artifacts directory before training.
    #[arg(short = 'c', long = "cleanup", default_value = "false")]
    pub cleanup: bool,
    /// A CSV file containing the dataset to train the model on.
    pub train_dataset: String,
    /// A CSV file containing the dataset to validate the model on.
    pub valid_dataset: String,
}

/// Arguments for the predict command.
#[derive(Args)]
pub struct PredictArgs {
    /// Path to the model configuration. Will be created if it doesn't exist yet.
    #[arg(short = 'm', long = "model", default_value = "./config/model.json")]
    pub model: String,
    /// The artifacts directory to use for training.
    #[arg(short = 'a', long = "artifacts", default_value = "./artifacts")]
    pub artifacts: String,
    /// The interval for the request.
    #[arg(short = 'i', long = "interval", default_value = "1d")]
    pub interval: String,
    /// The start date to predict the data from in `dd-MM-YYYY` format.
    pub start: String,
    /// The end date to predict the data from in `dd-MM-YYYY` format.
    pub end: String,
    /// The ticker to predict for.
    pub ticker: String,
    /// Other tickers to include in the features input.
    #[arg(value_delimiter = ',')]
    pub others: Vec<String>,
}

/// Arguments for the dump command.
#[derive(Args)]
pub struct DumpArgs {
    /// The data kind to dump. Available: 'database'.
    pub kind: String,
}
