// GPT-2スタイルのTransformerモデル実装
// 教育目的のシンプルな実装

pub mod working_transformer;

// 主要な型を再エクスポート
pub use working_transformer::{SimpleTokenizer, WorkingTransformer, TrainableTransformer};