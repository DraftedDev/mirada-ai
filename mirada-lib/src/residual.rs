use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::Backend;

#[derive(Debug, Module)]
pub struct ResidualBlock<B: Backend> {
    norm1: LayerNorm<B>,
    lin1: Linear<B>,
    lin2: Linear<B>,
    dropout: Dropout,
    norm2: LayerNorm<B>,
    act: Gelu,
}

impl<B: Backend> ResidualBlock<B> {
    pub fn forward(&self, input: burn::tensor::Tensor<B, 2>) -> burn::tensor::Tensor<B, 2> {
        let mut x = self.norm1.forward(input.clone());
        x = self.lin1.forward(x);
        x = self.act.forward(x);
        x = self.dropout.forward(x);
        x = self.lin2.forward(x);
        x = self.dropout.forward(x);

        self.norm2.forward(x + input)
    }
}

#[derive(Debug, Config)]
pub struct ResidualBlockConfig {
    /// Dimension of the hidden layer (input and output)
    pub hidden_dim: usize,

    /// Dropout probability
    pub dropout: f64,
}

impl ResidualBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResidualBlock<B> {
        ResidualBlock {
            norm1: LayerNormConfig::new(self.hidden_dim).init(device),
            lin1: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            lin2: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            norm2: LayerNormConfig::new(self.hidden_dim).init(device),
            act: Gelu,
        }
    }
}
