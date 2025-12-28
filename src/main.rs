use crate::cli::{Command, MirandaCli};
use clap::Parser;
use log::LevelFilter;

pub mod cli;

pub mod fetch;

pub mod train;

pub mod predict;

pub mod dump;

pub mod utils;

fn main() {
    env_logger::builder()
        .format_target(utils::env_or_default("LOG_TARGET", false))
        .format_file(utils::env_or_default("LOG_FILE", false))
        .format_line_number(utils::env_or_default("LOG_LINE_NUMBER", false))
        .format_module_path(utils::env_or_default("LOG_MODULE_PATH", false))
        .format_source_path(utils::env_or_default("LOG_SOURCE_PATH", false))
        .format_timestamp(None)
        .filter_level(utils::env_or_default("LOG_LEVEL", LevelFilter::Info))
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
            args.cleanup,
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
        Command::Dump(args) => dump::dump(cli.database, args.kind),
    }
}
