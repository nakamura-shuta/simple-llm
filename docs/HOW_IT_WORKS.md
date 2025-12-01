# 動作原理：具体例で理解するTransformer

このドキュメントでは、「Rust」と入力して「は プログラミング 言語 です」が生成されるまでの**全過程**を、コードと対応させて詳細に説明します。

---

## 目次

1. [全体の流れ](#全体の流れ)
2. [Phase 1: 学習（トレーニング）](#phase-1-学習トレーニング)
3. [Phase 2: テキスト生成](#phase-2-テキスト生成)
4. [なぜその単語が選ばれるのか](#なぜその単語が選ばれるのか)
5. [コード対応表](#コード対応表)

---

## 全体の流れ

```
┌─────────────────────────────────────────────────────────────┐
│                      学習フェーズ                            │
│  「Rust は プログラミング 言語 です」を学習                    │
│       ↓                                                     │
│  「Rust」→「は」の関係を記憶                                  │
│  「は」→「プログラミング」の関係を記憶                         │
│       ...                                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      生成フェーズ                            │
│  入力:「Rust」                                               │
│       ↓                                                     │
│  予測:「は」(確率80%)を選択                                   │
│       ↓                                                     │
│  予測:「プログラミング」(確率75%)を選択                        │
│       ...                                                   │
│  出力:「は プログラミング 言語 です」                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: 学習（トレーニング）

### 1.1 学習データの準備

**ファイル: `data.txt`**
```
Rust は プログラミング 言語 です
Rust は 高速 な 言語 です
Rust は 安全 な 言語 です
```

**初心者向け説明**: これが「教科書」です。モデルはこの文章を何度も読んで、パターンを覚えます。

---

### 1.2 トークン化（文字→数字への変換）

**コード: `src/working_transformer.rs` SimpleTokenizer**

```rust
// 語彙の構築（どの単語を知っているか）
pub fn build_vocab(&mut self, texts: &[&str]) {
    // 特殊トークンを追加
    self.add_token("<eos>");  // ID: 0 - 文の終わり
    self.add_token("<bos>");  // ID: 1 - 文の始まり
    self.add_token("<unk>");  // ID: 2 - 未知語

    // テキストから単語を抽出
    for text in texts {
        for word in text.split_whitespace() {
            self.add_token(word);
        }
    }
}
```

**処理の流れ:**

```
学習データ: "Rust は プログラミング 言語 です"
                    ↓
           スペースで分割
                    ↓
        ["Rust", "は", "プログラミング", "言語", "です"]
                    ↓
           各単語にIDを割り当て
                    ↓
┌────────────────┬─────┐
│ 単語            │ ID  │
├────────────────┼─────┤
│ <eos>          │  0  │
│ <bos>          │  1  │
│ <unk>          │  2  │
│ こんにちは      │  3  │
│ 世界           │  4  │
│ Rust           │  5  │  ← ここ！
│ ...            │ ... │
│ は             │ 10  │  ← ここ！
│ プログラミング  │ 11  │
│ 言語           │ 12  │
│ です           │ 13  │
└────────────────┴─────┘
```

**エンコード処理:**

```rust
// テキストをトークンIDに変換
pub fn encode(&self, text: &str) -> Vec<usize> {
    let mut tokens = vec![self.bos_token_id];  // [1] から開始

    for word in text.split_whitespace() {
        let id = self.token_to_id
            .get(word)
            .copied()
            .unwrap_or(self.unk_token_id);  // 知らない単語は<unk>
        tokens.push(id);
    }

    tokens.push(self.eos_token_id);  // [0] で終了
    tokens
}
```

**具体例:**
```
"Rust は プログラミング 言語 です"
            ↓ encode()
[1, 5, 10, 11, 12, 13, 0]
 ↑  ↑   ↑   ↑   ↑   ↑  ↑
BOS Rust は  プロ 言語 です EOS
```

---

### 1.3 埋め込み（数字→ベクトルへの変換）

**コード: `src/working_transformer.rs` WorkingTransformer**

```rust
// 埋め込み層
token_embeddings: Array2<f32>,    // [語彙サイズ × 隠れ次元] = [58 × 64]
position_embeddings: Array2<f32>, // [最大長 × 隠れ次元] = [128 × 64]
```

**処理の流れ:**

```
トークンID: [1, 5, 10, 11, 12, 13, 0]
                    ↓
        各IDに対応するベクトルを取得
                    ↓
┌─────────────────────────────────────────────────┐
│ ID=5 (Rust) のベクトル（64次元）:                 │
│ [0.12, -0.34, 0.56, ..., 0.78]                  │
│      (64個の数字)                                │
├─────────────────────────────────────────────────┤
│ ID=10 (は) のベクトル（64次元）:                  │
│ [-0.23, 0.45, -0.12, ..., 0.34]                 │
│      (64個の数字)                                │
└─────────────────────────────────────────────────┘
                    ↓
        位置情報を加算
                    ↓
最終埋め込み = トークン埋め込み + 位置埋め込み
```

**なぜベクトル？**
- 「Rust」と「Python」は似た意味（プログラミング言語）→ ベクトルも近い
- 「Rust」と「天気」は違う意味 → ベクトルは遠い
- 数字で表現することで、コンピュータが「意味」を計算できる

---

### 1.4 Transformerレイヤーの処理

**コード: `src/working_transformer.rs` transformer_layer()**

```rust
fn transformer_layer(&self, x: &Array2<f32>, layer_idx: usize) -> Array2<f32> {
    // 1. Layer Normalization（数値を安定化）
    let x_norm = self.layer_norm(x, layer_idx, true);

    // 2. Multi-Head Attention（どの単語に注目するか）
    let attn_output = self.multi_head_attention(&x_norm, layer_idx);

    // 3. 残差接続（元の情報を保持）
    let x = x + &attn_output;

    // 4. FFN（情報を変換）
    let x_norm2 = self.layer_norm(&x, layer_idx, false);
    let ffn_output = self.feed_forward(&x_norm2, layer_idx);

    // 5. 残差接続
    x + &ffn_output
}
```

#### Attention（注意機構）の仕組み

```
入力: "Rust は プログラミング"
        ↓
Q(Query): 「私は何を探している？」
K(Key):   「私は何を提供できる？」
V(Value): 「私の情報は何？」
        ↓
「プログラミング」を処理する時:
  - 「Rust」に注目度 0.6
  - 「は」に注目度 0.1
  - 「プログラミング」に注目度 0.3
        ↓
加重平均で情報を集約
```

**Causal Mask（未来を見ない）:**

```
           Rust  は  プロ  言語  です
Rust    [  1    0    0    0    0  ]  ← Rustは自分だけ見る
は      [  1    1    0    0    0  ]  ← 「は」はRustと自分を見る
プロ    [  1    1    1    0    0  ]  ← 「プロ」は前3つを見る
言語    [  1    1    1    1    0  ]
です    [  1    1    1    1    1  ]

1 = 見れる、0 = 見れない（マスク）
```

**コード:**
```rust
fn multi_head_attention(&self, x: &Array2<f32>, layer_idx: usize) -> Array2<f32> {
    // Q, K, V を計算
    let q = x.dot(&self.layers[layer_idx].w_q);
    let k = x.dot(&self.layers[layer_idx].w_k);
    let v = x.dot(&self.layers[layer_idx].w_v);

    // 注意スコアを計算
    let scores = q.dot(&k.t()) / (self.head_dim as f32).sqrt();

    // Causal maskを適用（未来を見ない）
    let masked_scores = self.apply_causal_mask(&scores);

    // Softmaxで確率に変換
    let attn_weights = self.softmax(&masked_scores);

    // 加重平均
    attn_weights.dot(&v).dot(&self.layers[layer_idx].w_o)
}
```

---

### 1.5 損失計算と学習

**学習の目標:** 「Rust」の次は「は」、「は」の次は「プログラミング」...と予測できるようになること

**コード: `src/working_transformer.rs` backward()**

```rust
// 学習データの準備
入力:   [BOS, Rust, は,   プロ,  言語]  // モデルへの入力
正解:   [Rust, は,  プロ, 言語,  です]  // 予測すべき正解
```

**損失計算（どれくらい間違えたか）:**

```
モデルの予測（各位置で次の単語の確率）:

位置0（BOSの後）:
  Rust: 0.3, は: 0.1, プログラミング: 0.05, ...
  正解は「Rust」→ 確率0.3は低い → 損失大

↓ 学習を繰り返すと ↓

位置0（BOSの後）:
  Rust: 0.8, は: 0.05, プログラミング: 0.02, ...
  正解は「Rust」→ 確率0.8は高い → 損失小
```

**重み更新（逆伝播）:**

```rust
pub fn train(&mut self, texts: &[String], epochs: usize) {
    for epoch in 0..epochs {
        for text in texts {
            // 1. 順伝播（予測を計算）
            let (logits, cache) = self.model.forward_full_cache(&input);

            // 2. 損失計算（間違いを測定）
            let loss = self.model.backward(&cache, &target, &mut grads);

            // 3. 重み更新（パラメータを調整）
            self.model.apply_gradients(&grads, lr, weight_decay);
        }
    }
}
```

**学習の進行:**
```
Epoch 1:   損失 = 4.39  （ほぼランダム）
Epoch 50:  損失 = 1.85  （パターンを学習中）
Epoch 100: 損失 = 0.98  （かなり学習）
Epoch 300: 損失 = 0.66  （よく学習した）
```

---

## Phase 2: テキスト生成

### 2.1 入力のトークン化

**ユーザー入力:** `Rust`

**コード: `src/working_transformer.rs` generate()**

```rust
pub fn generate(&self, prompt: &str, max_new_tokens: usize, temperature: f32) -> String {
    // 1. プロンプトをトークン化
    let mut tokens = self.tokenizer.encode(prompt);
    // "Rust" → [1, 5, 0] = [BOS, Rust, EOS]

    // EOSを除去（生成を続けるため）
    tokens.pop();
    // [1, 5] = [BOS, Rust]
```

**処理の流れ:**
```
入力: "Rust"
        ↓ encode()
[1, 5, 0]
 ↑  ↑  ↑
BOS Rust EOS
        ↓ EOSを除去
[1, 5]
 ↑  ↑
BOS Rust ← この続きを予測する
```

---

### 2.2 次の単語の予測（1回目）

**コード:**

```rust
// 2. モデルで次の単語を予測
let logits = self.model.forward(&tokens);
// logits: [語彙サイズ] = [58個の数値]

// 最後の位置のlogitsを取得（次の単語の予測）
let last_logit = logits.row(logits.nrows() - 1);
```

**処理の流れ:**

```
入力トークン: [1, 5] (BOS, Rust)
                ↓
        Transformerで処理
                ↓
出力 logits（各単語の「出やすさ」スコア）:

┌────────────────┬─────────┬────────────┐
│ 単語            │ logit   │ 意味       │
├────────────────┼─────────┼────────────┤
│ <eos>          │ -2.1    │ 出にくい   │
│ <bos>          │ -∞      │ マスク済み │
│ <unk>          │ -∞      │ マスク済み │
│ こんにちは      │ -1.5    │ 出にくい   │
│ Rust           │ -0.8    │ 少し出る   │
│ は             │  2.3    │ 出やすい！ │ ← 最高スコア
│ プログラミング  │  0.5    │ まあまあ   │
│ ...            │ ...     │            │
└────────────────┴─────────┴────────────┘
```

---

### 2.3 温度付きサンプリング

**コード:**

```rust
// 3. 温度を適用してSoftmax
let scaled_logits = last_logit.mapv(|x| x / temperature);  // temperature=0.3
let probs = self.softmax(&scaled_logits);

// 4. 確率に基づいてサンプリング
let next_token = self.sample_from_probs(&probs);
```

**温度の効果:**

```
温度 = 1.0（高い）の場合:
┌────────────────┬─────────┐
│ 単語            │ 確率    │
├────────────────┼─────────┤
│ は             │ 45%     │
│ プログラミング  │ 20%     │
│ Rust           │ 15%     │
│ その他         │ 20%     │
└────────────────┴─────────┘
→ 色々な単語が選ばれる可能性（創造的）

温度 = 0.3（低い）の場合:
┌────────────────┬─────────┐
│ 単語            │ 確率    │
├────────────────┼─────────┤
│ は             │ 92%     │ ← ほぼこれが選ばれる
│ プログラミング  │  5%     │
│ Rust           │  2%     │
│ その他         │  1%     │
└────────────────┴─────────┘
→ 最も確率の高い単語が選ばれる（安定的）
```

**結果:** 「は」が選ばれる（確率92%）

---

### 2.4 生成の繰り返し

**コード:**

```rust
    // 生成ループ
    for _ in 0..max_new_tokens {
        let logits = self.model.forward(&tokens);
        let probs = self.softmax(&scaled_logits);
        let next_token = self.sample_from_probs(&probs);

        // EOSが出たら終了
        if next_token == self.tokenizer.eos_token_id {
            break;
        }

        tokens.push(next_token);
        generated_tokens.push(next_token);
    }
```

**生成の進行:**

```
ステップ1:
  入力: [BOS, Rust]
  予測: 「は」(92%) → 選択
  結果: [BOS, Rust, は]

ステップ2:
  入力: [BOS, Rust, は]
  予測: 「プログラミング」(78%) → 選択
  結果: [BOS, Rust, は, プログラミング]

ステップ3:
  入力: [BOS, Rust, は, プログラミング]
  予測: 「言語」(85%) → 選択
  結果: [BOS, Rust, は, プログラミング, 言語]

ステップ4:
  入力: [BOS, Rust, は, プログラミング, 言語]
  予測: 「です」(90%) → 選択
  結果: [BOS, Rust, は, プログラミング, 言語, です]

ステップ5:
  入力: [BOS, Rust, は, プログラミング, 言語, です]
  予測: 「<eos>」(75%) → 終了
```

---

### 2.5 デコード（数字→文字への変換）

**コード:**

```rust
// 5. トークンIDをテキストに変換
pub fn decode(&self, tokens: &[usize]) -> String {
    tokens
        .iter()
        .filter(|&&t| t != self.bos_token_id && t != self.eos_token_id)
        .map(|&t| self.id_to_token[t].as_str())
        .collect::<Vec<_>>()
        .join(" ")
}
```

**処理の流れ:**
```
生成されたトークン: [10, 11, 12, 13]
                        ↓ decode()
                   各IDを単語に変換
                        ↓
                ["は", "プログラミング", "言語", "です"]
                        ↓ join(" ")
                "は プログラミング 言語 です"
```

---

## なぜその単語が選ばれるのか

### 学習データとの対応

```
学習データ:
  "Rust は プログラミング 言語 です"
  "Rust は 高速 な 言語 です"
  "Rust は 安全 な 言語 です"

モデルが学習したパターン:
  「Rust」の後 → 「は」が3回出現 → 確率高い
  「は」の後 → 「プログラミング」「高速」「安全」が各1回
             → 「プログラミング」「高速」「安全」が同確率
```

### Attentionの働き

```
「Rust は プログラミング」を処理中、「言語」を予測する時:

Attention重み:
  「Rust」への注目: 0.4  ← プログラミング言語の文脈
  「は」への注目: 0.1
  「プログラミング」への注目: 0.5  ← 直前の単語

「プログラミング」+「Rust」の文脈から
→ 「言語」が最も適切と判断
```

---

## コード対応表

| 処理 | ファイル | 関数/メソッド | 行番号(目安) |
|------|----------|--------------|-------------|
| トークン化 | `working_transformer.rs` | `SimpleTokenizer::encode()` | 492-510 |
| 語彙構築 | `working_transformer.rs` | `SimpleTokenizer::build_vocab()` | 482-495 |
| 埋め込み | `working_transformer.rs` | `WorkingTransformer::forward()` | 289-310 |
| Attention | `working_transformer.rs` | `multi_head_attention()` | 205-243 |
| FFN | `working_transformer.rs` | `feed_forward()` | 246-253 |
| LayerNorm | `working_transformer.rs` | `layer_norm()` | 256-272 |
| 生成 | `working_transformer.rs` | `TrainableTransformer::generate()` | 386-436 |
| 学習 | `working_transformer.rs` | `TrainableTransformer::train()` | 1168-1260 |
| 逆伝播 | `working_transformer.rs` | `WorkingTransformer::backward()` | 428-600 |

---

## まとめ：全体の流れ

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 学習フェーズ                                                  │
│    data.txt → トークン化 → Transformer → 損失計算 → 重み更新     │
│    （300エポック繰り返し）                                        │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. 生成フェーズ                                                  │
│                                                                 │
│    "Rust" → [1,5] → Transformer → logits → softmax → サンプル   │
│                                                    ↓            │
│                                                  「は」          │
│                                                    ↓            │
│    "Rust は" → [1,5,10] → Transformer → ... → 「プログラミング」 │
│                                                    ↓            │
│    ... 繰り返し ...                                              │
│                                                    ↓            │
│    出力: "は プログラミング 言語 です"                            │
└─────────────────────────────────────────────────────────────────┘
```

**ポイント:**
1. **学習**: 「Rust」の後に「は」が来るパターンを記憶
2. **埋め込み**: 単語を数値ベクトルに変換
3. **Attention**: 文脈（前の単語）を考慮して予測
4. **温度**: 生成の「確実さ」を調整
5. **自己回帰**: 1単語ずつ順番に生成
