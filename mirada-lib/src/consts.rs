pub const WINDOW_Z: usize = 90;
pub const WINDOW_SCALE: usize = 252;
pub const CLIP: f32 = 6.0;
pub const HORIZON: usize = 10;
pub const FEATURE_SIZE: usize = 18;
pub const OTHER_STOCKS: usize = 5;
pub const TOTAL_FEATURE_SIZE: usize = FEATURE_SIZE * (OTHER_STOCKS + 1);
pub const EPS: f32 = 1e-8;

/// Skipped timesteps equal to the largest window of the [feature_args] constants.
pub const SKIPPED_TIMESTEPS: usize = feature_args::VOL_Z_SC_WINDOW;

pub mod feature_args {
    pub const VOL_Z_SC_WINDOW: usize = 30;
    pub const ROLL_VOLAT_WINDOW: usize = 15;
    pub const VOLAT_RAT_SHORT: usize = 15;
    pub const VOLAT_RAT_LONG: usize = 30;
    pub const SMA_DIST_WINDOW: usize = 20;
    pub const EMA_DIST_SPAN: usize = 20;
    pub const PR_VS_LONG_SMA_WINDOW: usize = 20;
    pub const MOM_PERIOD: usize = 12;
    pub const ATR_PERIOD: usize = 14;
    pub const BOLL_BAND_WINDOW: usize = 20;
    pub const BOLL_BAND_NUM_STD: f32 = 2.0;
    pub const RSI_PERIOD: usize = 14;
    pub const MACD_HIST_SHORT: usize = 12;
    pub const MACD_HIST_LONG: usize = 16;
    pub const MACD_HIST_SIGNAL: usize = 9;
}

pub mod feature_index {
    pub const LOG_RET_IDX: usize = 0;
    pub const VOL_CHANGE_IDX: usize = 1;
    pub const VOL_U_SC_IDX: usize = 2;
    pub const PR_VOL_PRE_IDX: usize = 3;
    pub const ROLL_VOLAT_IDX: usize = 4;
    pub const VOLAT_RAT_IDX: usize = 5;
    pub const HL_RANGE_IDX: usize = 6;
    pub const OC_RET_IDX: usize = 7;
    pub const SMA_DIST_IDX: usize = 8;
    pub const EMA_DIST_IDX: usize = 9;
    pub const PR_VS_LONG_SMA_IDX: usize = 10;
    pub const MOM_IDX: usize = 11;
    pub const TRUE_RANGES_IDX: usize = 12;
    pub const ATR_IDX: usize = 13;
    pub const ATR_RAT_IDX: usize = 14;
    pub const BOLL_BAND_IDX: usize = 15;
    pub const RSI_IDX: usize = 16;
    pub const MACD_HIST_IDX: usize = 17;
}
