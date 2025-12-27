use crate::data::{StockData, TOTAL_FEATURE_SIZE};
use crate::model::Model;
use burn::prelude::Backend;

impl<B: Backend> Model<B> {
    pub fn infer(&self, input: StockData, device: &B::Device) -> f32 {
        let (features, _) = input.into_tensors(device);
        let shape = features.shape();

        let t = shape.dims[0];

        assert!(t > 0, "No timesteps available");
        assert_eq!(
            shape.dims[1], TOTAL_FEATURE_SIZE,
            "Expected TOTAL_FEATURE_SIZE = {TOTAL_FEATURE_SIZE}"
        );

        let input = features.slice([t - 1..t, 0..TOTAL_FEATURE_SIZE]);

        let output = self.forward(input).detach();

        output.into_data().to_vec::<f32>().unwrap()[0]
    }

    pub fn infer_price(&self, input: StockData, device: &B::Device) -> f32 {
        input.last_close() * self.infer(input, device).exp()
    }
}
