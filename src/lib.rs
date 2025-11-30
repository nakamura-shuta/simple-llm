// GPT-2スタイルのTransformerモデル実装
// 教育目的のシンプルな実装（完全な逆伝播対応版）

pub mod working_transformer;

// 主要な型を再エクスポート
pub use working_transformer::{
    SimpleTokenizer, TrainableTransformer, TrainingConfig, WorkingTransformer,
};