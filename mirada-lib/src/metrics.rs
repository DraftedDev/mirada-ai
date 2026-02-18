use crate::output::ModelOutput;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::metric::*;

#[derive(Clone, Default)]
pub struct SharpeRatioMetrics<B: Backend> {
    returns: Vec<f64>,
    state: NumericMetricState,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Metric for SharpeRatioMetrics<B> {
    type Input = ProfitInput<B>;

    fn name(&self) -> MetricName {
        MetricName::new("Sharpe Ratio".to_string())
    }

    fn attributes(&self) -> MetricAttributes {
        MetricAttributes::Numeric(NumericAttributes {
            unit: None,
            higher_is_better: true,
        })
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let preds = item.logits.clone().argmax(1);

        let preds: Vec<i32> = preds.into_data().to_vec::<i32>().unwrap();
        let rets: Vec<f32> = item.returns.clone().into_data().to_vec::<f32>().unwrap();

        // Convert per-trade to signed returns based on prediction
        let batch_returns: Vec<f64> = preds
            .iter()
            .zip(rets.iter())
            .map(|(p, r)| if *p == 1 { *r as f64 } else { -*r as f64 })
            .collect();

        self.returns.extend(batch_returns);

        // Compute running Sharpe: mean / std-dev
        let n = self.returns.len();
        let mean = self.returns.iter().sum::<f64>() / n as f64;
        let variance = self
            .returns
            .iter()
            .map(|x| (*x - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let std_dev = variance.sqrt().max(1e-8); // avoid div0
        let sharpe = mean / std_dev;

        let [batch_size] = item.returns.dims();

        self.state.update(
            sharpe,
            batch_size,
            FormatOptions::new(MetricName::new("sharpe ratio".to_string())).precision(2),
        )
    }

    fn clear(&mut self) {
        self.returns.clear();
        self.state.reset();
    }
}

impl<B: Backend> Numeric for SharpeRatioMetrics<B> {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

pub struct ProfitInput<B: Backend> {
    pub logits: Tensor<B, 2>,
    pub returns: Tensor<B, 1>,
}

impl<B: Backend> Adaptor<ProfitInput<B>> for ModelOutput<B> {
    fn adapt(&self) -> ProfitInput<B> {
        ProfitInput {
            logits: self.output.clone(),
            returns: self.returns.clone(),
        }
    }
}
