use burn::Tensor;
use burn::prelude::{Backend, Int};
use burn::tensor::Transaction;
use burn::train::ItemLazy;
use burn::train::metric::{
    AccuracyInput, Adaptor, ConfusionStatsInput, LossInput, PerplexityInput, TopKAccuracyInput,
};

pub struct ModelOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub output: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
    pub returns: Tensor<B, 1>,
}

impl<B: Backend> ItemLazy for ModelOutput<B> {
    type ItemSync = ModelOutput<B>;

    fn sync(self) -> Self::ItemSync {
        let [output, loss, targets, returns] = Transaction::default()
            .register(self.output)
            .register(self.loss)
            .register(self.targets)
            .register(self.returns)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &Default::default();

        ModelOutput {
            output: Tensor::from_data(output, device),
            loss: Tensor::from_data(loss, device),
            targets: Tensor::from_data(targets, device),
            returns: Tensor::from_data(returns, device),
        }
    }
}

impl<B: Backend> Adaptor<AccuracyInput<B>> for ModelOutput<B> {
    fn adapt(&self) -> AccuracyInput<B> {
        AccuracyInput::new(self.output.clone(), self.targets.clone())
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for ModelOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: Backend> Adaptor<TopKAccuracyInput<B>> for ModelOutput<B> {
    fn adapt(&self) -> TopKAccuracyInput<B> {
        TopKAccuracyInput::new(self.output.clone(), self.targets.clone())
    }
}

impl<B: Backend> Adaptor<PerplexityInput<B>> for ModelOutput<B> {
    fn adapt(&self) -> PerplexityInput<B> {
        PerplexityInput::new(self.output.clone(), self.targets.clone())
    }
}

impl<B: Backend> Adaptor<ConfusionStatsInput<B>> for ModelOutput<B> {
    fn adapt(&self) -> ConfusionStatsInput<B> {
        let [_, num_classes] = self.output.dims();
        if num_classes > 1 {
            ConfusionStatsInput::new(
                self.output.clone(),
                self.targets.clone().one_hot(num_classes).bool(),
            )
        } else {
            ConfusionStatsInput::new(
                self.output.clone(),
                self.targets.clone().unsqueeze_dim(1).bool(),
            )
        }
    }
}
