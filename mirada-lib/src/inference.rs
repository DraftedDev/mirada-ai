use crate::consts::TOTAL_FEATURE_SIZE;
use crate::data::StockData;
use crate::model::Model;
use burn::prelude::Backend;

impl<B: Backend> Model<B> {
    /// Infer the class of the given input.
    ///
    /// See [crate::math::generate_targets] for info about classes.
    pub fn infer(&self, input: StockData, device: &B::Device) -> i32 {
        let (features, _, _) = input.into_tensors(device);

        let t = features.dims()[0];

        assert!(t > 0, "No timesteps available");
        assert_eq!(
            features.dims()[1],
            TOTAL_FEATURE_SIZE,
            "Expected TOTAL_FEATURE_SIZE = {TOTAL_FEATURE_SIZE}"
        );

        let input = features.slice([t - 1..t, 0..TOTAL_FEATURE_SIZE]);

        let output = self.forward(input).detach();

        // Argmax along class dimension (2 classes)
        let preds = output
            .argmax(1)
            .into_data()
            .to_vec::<i32>()
            .expect("Failed to convert output to vector");

        preds[0]
    }
}
