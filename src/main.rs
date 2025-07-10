use anyhow::Result;
use simple_llm::TrainableTransformer;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::env;

/// ファイルから行を読み込むヘルパー関数
fn read_lines<P>(filename: P) -> io::Result<Vec<String>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    let buf = BufReader::new(file);
    buf.lines().collect()
}

/// トレーニングループ（実際の学習）
fn train_model(
    model: &mut TrainableTransformer,
    texts: &[String],
    epochs: usize,
) {
    println!("トレーニング開始...");
    println!("エポック数: {}", epochs);
    println!("データセットサイズ: {} 文", texts.len());
    println!();

    // 実際の学習を実行
    model.train(texts, epochs);
    
    println!("\nトレーニング完了！");
}

/// インタラクティブな生成モード
fn interactive_generation(model: &TrainableTransformer) {
    println!("\n=== テキスト生成モード ===");
    println!("プロンプトを入力してください (終了: 'quit' または 'exit')");
    println!("生成パラメータ: temperature=0.8, 単語単位生成");
    println!();

    let stdin = io::stdin();
    loop {
        print!("> ");
        io::Write::flush(&mut io::stdout()).unwrap();

        let mut input = String::new();
        stdin.read_line(&mut input).unwrap();
        let input = input.trim();

        if input == "quit" || input == "exit" {
            break;
        }

        if input.is_empty() {
            continue;
        }

        // テキスト生成
        let max_words = if input.contains("質問") && input.contains("回答") {
            20  // Q&A形式の場合は長めに生成
        } else {
            1   // 通常は1単語
        };
        
        let generated = model.generate(
            input,
            max_words,
            0.8,        // temperature
        );

        println!("\n生成結果:");
        println!("{}", generated);
        println!();
    }
}

fn main() -> Result<()> {
    println!("=== GPT-2 教育用実装 (Rust) ===\n");

    // コマンドライン引数から言語を取得
    let args: Vec<String> = env::args().collect();
    let lang = if args.len() > 1 && args[1] == "en" {
        "en"
    } else {
        "ja"
    };

    // データセットの読み込み
    let data_path = if lang == "en" {
        "data_en.txt"
    } else {
        "data.txt"
    };
    
    println!("言語: {} (データファイル: {})", 
             if lang == "en" { "英語" } else { "日本語" }, 
             data_path);
    
    let texts = match read_lines(data_path) {
        Ok(lines) => lines,
        Err(e) => {
            eprintln!("データファイル '{}' の読み込みに失敗しました: {}", data_path, e);
            if lang == "en" {
                eprintln!("data_en.txt ファイルを作成し、英語のトレーニングテキストを追加してください。");
            } else {
                eprintln!("data.txt ファイルを作成し、日本語のトレーニングテキストを追加してください。");
            }
            eprintln!("\n使い方:");
            eprintln!("  日本語: cargo run");
            eprintln!("  英語: cargo run -- en");
            return Ok(());
        }
    };

    // モデルの初期化（トークナイザーを含む）
    println!("\nモデルを初期化中...");
    
    // モデル設定
    let hidden_size = 64;
    let num_heads = 4;
    let num_layers = 2;
    let max_seq_len = 128;
    let learning_rate = 0.01;  // 学習率を上げて効果を見やすくする
    
    // テキストデータからモデルを作成
    let texts_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let mut model = TrainableTransformer::from_texts(
        &texts_refs,
        hidden_size,
        num_heads,
        num_layers,
        max_seq_len,
        learning_rate,
    );
    
    println!("モデル設定:");
    println!("  - 語彙サイズ: {}", model.tokenizer.vocab_size());
    println!("  - 埋め込み次元: {}", hidden_size);
    println!("  - ヘッド数: {}", num_heads);
    println!("  - レイヤー数: {}", num_layers);
    println!("  - 最大シーケンス長: {}", max_seq_len);
    println!("  - 学習率: {}", learning_rate);

    // トレーニング（エポック数を増やす）
    train_model(&mut model, &texts, 100);

    // インタラクティブ生成
    interactive_generation(&model);

    println!("\nプログラム終了");
    Ok(())
}