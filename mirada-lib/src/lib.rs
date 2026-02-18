pub mod batcher;
pub mod consts;
pub mod data;
pub mod database;
pub mod dataset;
pub mod eval;
pub mod inference;
pub mod math;
pub mod metrics;
pub mod model;
pub mod output;
pub mod training;

#[cfg(feature = "cuda")]
pub type Backend = burn::backend::Cuda;

#[cfg(feature = "cuda")]
pub type Device = burn::backend::cuda::CudaDevice;

pub type AutodiffBackend = burn::backend::Autodiff<Backend>;
