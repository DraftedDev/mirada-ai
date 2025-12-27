use crate::cli::{Command, MirandaCli};
use clap::Parser;
use log::LevelFilter;
use std::str::FromStr;

pub mod cli;

pub mod fetch;

pub mod train;

pub mod predict;

pub mod data_dump;

pub mod utils;

fn main() {
    // TODO: config via env
    env_logger::builder()
        .format_target(false)
        .format_file(false)
        .format_line_number(false)
        .format_module_path(true)
        .format_source_path(false)
        .format_timestamp(None)
        .filter_level(
            LevelFilter::from_str(
                std::env::var("LOG_LEVEL")
                    .unwrap_or("info".to_string())
                    .as_str(),
            )
            .expect("Invalid log level"),
        )
        .filter_module("cubecl_cuda", LevelFilter::Error)
        .filter_module("burn", LevelFilter::Error)
        .init();

    let cli = MirandaCli::parse();

    match cli.command {
        Command::Fetch(args) => fetch::fetch(
            cli.database,
            cli.timeout,
            args.interval,
            args.start,
            args.end,
            args.ticker,
            args.file,
        ),
        Command::Train(args) => train::train(
            cli.database,
            args.train_dataset,
            args.valid_dataset,
            args.model,
            args.training,
            args.interval,
            args.artifacts,
        ),
        Command::Predict(args) => predict::predict(
            cli.timeout,
            args.artifacts,
            args.model,
            args.interval,
            args.start,
            args.end,
            args.ticker,
            args.others,
        ),
        Command::DataDump => data_dump::data_dump(cli.database),
    }
}
