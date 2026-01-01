use burn::Tensor;
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::Backend;

#[derive(Debug, Module)]
pub struct ResidualBlock<B: Backend> {
    linear1: Linear<B>,
    activation: Gelu,
    linear2: Linear<B>,
    norm: LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> ResidualBlock<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let input_norm = self.norm.forward(input.clone());

        let mut x = self.linear1.forward(input_norm);

        x = self.activation.forward(x);
        x = self.linear2.forward(x);
        x = self.dropout.forward(x);

        // Residual connection
        x + input
    }
}

#[derive(Copy, Debug, Config)]
pub struct ResidualBlockConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub dropout_prob: f64,
}

impl ResidualBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResidualBlock<B> {
        ResidualBlock {
            linear1: LinearConfig::new(self.input_size, self.hidden_size).init(device),
            activation: Gelu,
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            norm: LayerNormConfig::new(self.input_size).init(device),
            dropout: DropoutConfig::new(self.dropout_prob).init(),
        }
    }
}
