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
    #[arg(short = 't', long = "timeout", default_value = "12000")]
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
    /// Evaluates a model on the given dataset.
    Eval(EvalArgs),
    /// Dumps the given data kind out to the console.
    Dump(DumpArgs),
    /// CSV generation utilities.
    Csv(CsvArgs),
}

/// Arguments for the fetch command.
#[derive(Args)]
pub struct FetchArgs {
    /// Whether to fetch data one by one instead of parallel using multiple threads.
    #[arg(short = 's', long = "serial", default_value_t = false)]
    pub serial: bool,
    /// How many times to retry failed requests. Set to 0 to disable.
    #[arg(short = 'r', long = "retry", default_value_t = 5)]
    pub retry: u8,
    /// Override and don't skip existing data.
    #[arg(short = 'o', long = "override", default_value_t = false)]
    pub _override: bool,
    /// The file to use to get fetching information. Generated via `csv fetch`.
    pub file: String,
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

/// Arguments for the eval command.
#[derive(Args)]
pub struct EvalArgs {
    /// Path to the model configuration. Will be created if it doesn't exist yet.
    #[arg(short = 'm', long = "model", default_value = "./config/model.json")]
    pub model: String,
    /// Path to the eval configuration. Will be created if it doesn't exist yet.
    #[arg(short = 'e', long = "eval", default_value = "./config/eval.json")]
    pub eval: String,
    /// The artifacts directory to use for training.
    #[arg(short = 'a', long = "artifacts", default_value = "./artifacts")]
    pub artifacts: String,
    /// A CSV file containing the dataset to evaluate the model on.
    pub dataset: String,
}

/// Arguments for the dump command.
#[derive(Args)]
pub struct DumpArgs {
    /// The data kind to dump. Available: 'database'.
    pub kind: String,
}

/// Arguments for the csv command.
#[derive(Args)]
pub struct CsvArgs {
    /// The subcommand to the csv command.
    #[command(subcommand)]
    pub command: CsvCommand,
}

/// The csv subcommand.
#[derive(Subcommand)]
pub enum CsvCommand {
    /// Generate CSV data to use as input for the `fetch -f` command.
    Fetch(CsvFetchArgs),
    /// Generate CSV data to use as input datasets for the `train` command.
    Train(CsvTrainArgs),
}

/// Arguments for the csv fetch command.
#[derive(Args)]
pub struct CsvFetchArgs {
    /// The output file to write to. Use `stdout` to write to the console.
    #[arg(short = 'o', long = "out", default_value = "stdout")]
    pub out: String,
    /// The start date to use for generation.
    pub start: String,
    /// The end date to use for generation.
    pub end: String,
    /// The length for each sample in days.
    pub length: u64,
    /// The range of shifts to use.
    pub shift: u64,
    /// The start of the jitter.
    pub jitter_start: u64,
    /// The end of the jitter.
    pub jitter_end: u64,
    /// The tickers to use for generation.
    #[arg(value_delimiter = ',', required = true)]
    pub tickers: Vec<String>,
}

/// Arguments for the csv train command.
#[derive(Args)]
pub struct CsvTrainArgs {
    /// The first output file to write to. Use `stdout` to write to the console.
    #[arg(short = '1', long = "out1", default_value = "stdout")]
    pub out1: String,
    /// The second output file to write to. Use `stdout` to write to the console.
    #[arg(short = '2', long = "out2", default_value = "stdout")]
    pub out2: String,
    /// The percentage of the input file to write to the first output file.
    pub percent: f32,
    /// The path to the input file to process.
    pub input: String,
}
