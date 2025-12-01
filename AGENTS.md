# Repository Guidelines

## プロジェクト構成とモジュール
- `src/main.rs`: CLIエントリーポイント。学習実行と対話生成を統括。
- `src/lib.rs`: ライブラリエクスポートと共通型定義。
- `src/working_transformer.rs`: GPT-2風Transformer本体（Attention、FFN、LayerNormなど）。
- `docs/`: 実装フローや比較資料。開発前に`PROCESSING_FLOW.md`を一読推奨。
- `data.txt` / `data_en.txt`: サンプル学習データ（1行1文）。追加データも同形式で配置。

## ビルド・テスト・開発コマンド
- `cargo build --release`: 最適化ビルド。配布・長時間学習時に使用。
- `cargo run` / `cargo run -- en`: 日本語/英語データで学習→対話モード開始。
- `cargo test`: 単体テスト実行（追加時）。失敗時は出力ログを共有。
- `cargo fmt`: rustfmtで整形。コミット前に必須。
- `cargo clippy -D warnings`: 静的検査。警告をエラーとして扱い、無視しない。

## コーディングスタイル・命名
- Rust標準スタイルを`cargo fmt`で統一。インデントは4スペース。
- 変数・関数は`snake_case`、型・構造体は`PascalCase`。
- モジュール分割: トークナイザ、学習ループ、モデルは役割ごとに関数化し`src/`内で整理。
- コメントは処理意図や数式の要点のみ簡潔に記述。

## テスト指針
- 期待値のある小規模テンプレートで関数テストを追加。ファイル名は`*_test.rs`またはモジュール内`mod tests`。
- 学習ループはステップ数を小さくしたスモークテストを用意し、実行時間を短く保つ。
- カバレッジ目安: 主要変換処理（トークナイザ、前向き計算）を優先して網羅。

## コミットとプルリク
- 現状履歴は`init`のみで明確な規約なし。今後はConventional Commits（例: `feat: add tokenizer tests`）を推奨。
- コミットは小さく論理単位で分割し、`cargo fmt`/`clippy`/`test`通過を確認。
- PRには目的、主要変更点、動作確認コマンド、関連Issueを記載。学習ログやスクリーンショットがあれば添付。

## セキュリティ・設定
- 学習データに個人情報を含めない。共有時はダミーデータに置換。
- 大規模データや長時間学習は`--release`と適切なバッチサイズを設定し、メモリ使用を監視。
- 環境差異を避けるためRustは最新stableを使用（`rustup update stable`推奨）。
