// 動作するTransformerモデルの実装

use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::HashMap;

/// 実際に動作するTransformerモデル
pub struct WorkingTransformer {
    // モデル設定
    vocab_size: usize,
    hidden_size: usize,
    num_heads: usize,
    num_layers: usize,
    max_seq_len: usize,
    head_dim: usize,
    
    // 埋め込み層
    token_embeddings: Array2<f32>,
    position_embeddings: Array2<f32>,
    
    // 各レイヤーのパラメータ
    layers: Vec<TransformerLayer>,
    
    // 最終層のLayer Norm
    ln_final_gamma: Array1<f32>,
    ln_final_beta: Array1<f32>,
    
    // 出力層
    output_projection: Array2<f32>,
    
    // 学習用の勾配（簡易版）
    token_embedding_grads: Array2<f32>,
    output_projection_grads: Array2<f32>,
}

/// Transformerの1レイヤー
struct TransformerLayer {
    // Multi-Head Attention
    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    w_o: Array2<f32>,
    
    // Layer Norm 1 (Attention前)
    ln1_gamma: Array1<f32>,
    ln1_beta: Array1<f32>,
    
    // Feed Forward
    ff_w1: Array2<f32>,
    ff_b1: Array1<f32>,
    ff_w2: Array2<f32>,
    ff_b2: Array1<f32>,
    
    // Layer Norm 2 (FFN前)
    ln2_gamma: Array1<f32>,
    ln2_beta: Array1<f32>,
}

impl WorkingTransformer {
    /// 新しいTransformerモデルを作成
    pub fn new(
        vocab_size: usize,
        hidden_size: usize,
        num_heads: usize,
        num_layers: usize,
        max_seq_len: usize,
    ) -> Self {
        assert!(hidden_size % num_heads == 0, "hidden_sizeはnum_headsで割り切れる必要があります");
        
        let head_dim = hidden_size / num_heads;
        let mut rng = rand::thread_rng();
        let init_scale = (1.0 / hidden_size as f32).sqrt();
        
        // 埋め込み層の初期化
        let token_embeddings = Array2::from_shape_fn((vocab_size, hidden_size), |_| {
            rng.gen_range(-init_scale..init_scale)
        });
        
        let position_embeddings = Array2::from_shape_fn((max_seq_len, hidden_size), |_| {
            rng.gen_range(-init_scale..init_scale)
        });
        
        // 各レイヤーの初期化
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(TransformerLayer::new(hidden_size, num_heads, &mut rng));
        }
        
        // 最終Layer Norm
        let ln_final_gamma = Array1::ones(hidden_size);
        let ln_final_beta = Array1::zeros(hidden_size);
        
        // 出力投影（token_embeddingsの転置）
        let output_projection = token_embeddings.t().to_owned();
        
        // 勾配用の配列を初期化
        let token_embedding_grads = Array2::zeros((vocab_size, hidden_size));
        let output_projection_grads = Array2::zeros((hidden_size, vocab_size));
        
        Self {
            vocab_size,
            hidden_size,
            num_heads,
            num_layers,
            max_seq_len,
            head_dim,
            token_embeddings,
            position_embeddings,
            layers,
            ln_final_gamma,
            ln_final_beta,
            output_projection,
            token_embedding_grads,
            output_projection_grads,
        }
    }
    
    /// 順伝播
    pub fn forward(&self, input_ids: &[usize], _training: bool) -> Vec<Array1<f32>> {
        let seq_len = input_ids.len();
        let _batch_size = 1; // 簡単のため
        
        // 埋め込み
        let mut hidden_states = Array2::zeros((seq_len, self.hidden_size));
        for (i, &token_id) in input_ids.iter().enumerate() {
            let token_emb = self.token_embeddings.row(token_id);
            let pos_emb = self.position_embeddings.row(i);
            let combined = &token_emb.to_owned() + &pos_emb.to_owned();
            hidden_states.row_mut(i).assign(&combined);
        }
        
        // 各Transformerレイヤーを通す
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states);
        }
        
        // 最終Layer Norm
        hidden_states = self.layer_norm(&hidden_states, &self.ln_final_gamma, &self.ln_final_beta);
        
        // 出力投影
        let mut logits = Vec::new();
        for i in 0..seq_len {
            let hidden = hidden_states.row(i);
            let logit = hidden.dot(&self.output_projection);
            logits.push(logit.to_owned());
        }
        
        logits
    }
    
    /// Layer Normalization
    fn layer_norm(&self, x: &Array2<f32>, gamma: &Array1<f32>, beta: &Array1<f32>) -> Array2<f32> {
        let (seq_len, hidden_size) = x.dim();
        let mut output = Array2::zeros((seq_len, hidden_size));
        
        for i in 0..seq_len {
            let row = x.row(i);
            let mean = row.mean().unwrap();
            let var = row.mapv(|v| (v - mean).powi(2)).mean().unwrap();
            let std = (var + 1e-5).sqrt();
            
            let normalized = row.mapv(|v| (v - mean) / std);
            let scaled = &normalized * gamma + beta;
            output.row_mut(i).assign(&scaled);
        }
        
        output
    }
}

impl TransformerLayer {
    fn new(hidden_size: usize, _num_heads: usize, rng: &mut impl Rng) -> Self {
        let scale = (1.0 / hidden_size as f32).sqrt();
        let ffn_hidden = hidden_size * 4;
        
        Self {
            // Attention weights
            w_q: Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.gen_range(-scale..scale)),
            w_k: Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.gen_range(-scale..scale)),
            w_v: Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.gen_range(-scale..scale)),
            w_o: Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.gen_range(-scale..scale)),
            
            // Layer Norm 1
            ln1_gamma: Array1::ones(hidden_size),
            ln1_beta: Array1::zeros(hidden_size),
            
            // Feed Forward
            ff_w1: Array2::from_shape_fn((hidden_size, ffn_hidden), |_| rng.gen_range(-scale..scale)),
            ff_b1: Array1::zeros(ffn_hidden),
            ff_w2: Array2::from_shape_fn((ffn_hidden, hidden_size), |_| rng.gen_range(-scale..scale)),
            ff_b2: Array1::zeros(hidden_size),
            
            // Layer Norm 2
            ln2_gamma: Array1::ones(hidden_size),
            ln2_beta: Array1::zeros(hidden_size),
        }
    }
    
    /// レイヤーの順伝播
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let (_seq_len, _hidden_size) = x.dim();
        
        // Attention部分
        let residual1 = x;
        let x_norm1 = self.layer_norm(x, &self.ln1_gamma, &self.ln1_beta);
        let attn_output = self.multi_head_attention(&x_norm1);
        let x = residual1 + &attn_output;
        
        // Feed Forward部分
        let residual2 = &x;
        let x_norm2 = self.layer_norm(&x, &self.ln2_gamma, &self.ln2_beta);
        let ffn_output = self.feed_forward(&x_norm2);
        residual2 + &ffn_output
    }
    
    /// Multi-Head Attention
    fn multi_head_attention(&self, x: &Array2<f32>) -> Array2<f32> {
        let (seq_len, hidden_size) = x.dim();
        let num_heads = hidden_size / (hidden_size / 8); // 仮に8ヘッド
        let head_dim = hidden_size / num_heads;
        
        // Q, K, Vの計算
        let q = x.dot(&self.w_q);
        let k = x.dot(&self.w_k);
        let v = x.dot(&self.w_v);
        
        // 簡易的なアテンション（実際はマルチヘッドに分割すべき）
        let scores = q.dot(&k.t()) / (head_dim as f32).sqrt();
        
        // Causal mask
        let mut attn_weights = Array2::zeros((seq_len, seq_len));
        for i in 0..seq_len {
            for j in 0..=i {
                let score = scores[[i, j]];
                attn_weights[[i, j]] = score;
            }
            
            // Softmax (各行に対して)
            let max_score = attn_weights.row(i).fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_sum = 0.0;
            for j in 0..=i {
                let exp_val = (attn_weights[[i, j]] - max_score).exp();
                attn_weights[[i, j]] = exp_val;
                exp_sum += exp_val;
            }
            for j in 0..=i {
                attn_weights[[i, j]] /= exp_sum;
            }
        }
        
        // Attention output
        let attn_output = attn_weights.dot(&v);
        attn_output.dot(&self.w_o)
    }
    
    /// Feed Forward Network
    fn feed_forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // 第1層 + GELU
        let h = x.dot(&self.ff_w1) + &self.ff_b1;
        let h_gelu = h.mapv(|v| gelu(v));
        
        // 第2層
        h_gelu.dot(&self.ff_w2) + &self.ff_b2
    }
    
    /// Layer Normalization
    fn layer_norm(&self, x: &Array2<f32>, gamma: &Array1<f32>, beta: &Array1<f32>) -> Array2<f32> {
        let (seq_len, hidden_size) = x.dim();
        let mut output = Array2::zeros((seq_len, hidden_size));
        
        for i in 0..seq_len {
            let row = x.row(i);
            let mean = row.mean().unwrap();
            let var = row.mapv(|v| (v - mean).powi(2)).mean().unwrap();
            let std = (var + 1e-5).sqrt();
            
            let normalized = row.mapv(|v| (v - mean) / std);
            let scaled = &normalized * gamma + beta;
            output.row_mut(i).assign(&scaled);
        }
        
        output
    }
}

/// GELU活性化関数
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// 学習可能なTransformerモデル
pub struct TrainableTransformer {
    model: WorkingTransformer,
    pub tokenizer: SimpleTokenizer,
    learning_rate: f32,
}

impl TrainableTransformer {
    pub fn new(
        vocab_size: usize,
        hidden_size: usize,
        num_heads: usize,
        num_layers: usize,
        max_seq_len: usize,
        learning_rate: f32,
    ) -> Self {
        let model = WorkingTransformer::new(vocab_size, hidden_size, num_heads, num_layers, max_seq_len);
        let tokenizer = SimpleTokenizer::new();
        
        Self {
            model,
            tokenizer,
            learning_rate,
        }
    }
    
    /// データから初期化
    pub fn from_texts(
        texts: &[&str],
        hidden_size: usize,
        num_heads: usize,
        num_layers: usize,
        max_seq_len: usize,
        learning_rate: f32,
    ) -> Self {
        let mut tokenizer = SimpleTokenizer::new();
        tokenizer.build_vocab(texts);
        
        let vocab_size = tokenizer.vocab_size();
        let model = WorkingTransformer::new(vocab_size, hidden_size, num_heads, num_layers, max_seq_len);
        
        Self {
            model,
            tokenizer,
            learning_rate,
        }
    }
    
    /// 簡易的な学習（デモ用）
    pub fn train(&mut self, texts: &[String], epochs: usize) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut count = 0;
            
            for text in texts {
                let tokens = self.tokenizer.encode(text);
                if tokens.len() < 2 {
                    continue;
                }
                
                // 入力とターゲット
                let input = &tokens[..tokens.len()-1];
                let target = &tokens[1..];
                
                // 順伝播
                let logits = self.model.forward(input, true);
                
                // 損失計算（簡易版）
                for (i, logit) in logits.iter().enumerate() {
                    let target_id = target[i];
                    let probs = softmax(logit);
                    let loss = -probs[target_id].ln();
                    total_loss += loss;
                    count += 1;
                    
                    // 簡易的な重み更新（実際は逆伝播が必要）
                    self.simple_weight_update(input[i], target_id);
                }
            }
            
            if count > 0 {
                println!("Epoch {}: 平均損失 = {:.4}", epoch + 1, total_loss / count as f32);
            }
        }
    }
    
    /// 簡易的な逆伝播と重み更新
    fn simple_weight_update(&mut self, input_id: usize, target_id: usize) {
        // クロスエントロピー損失の勾配を計算
        let logits = self.model.forward(&[input_id], true);
        if let Some(output) = logits.get(0) {
            // Softmax確率を計算
            let probs = softmax(output);
            
            // 勾配を計算（正解ラベルの確率から1を引く）
            let mut grad = probs.clone();
            grad[target_id] -= 1.0;
            
            // 出力層の勾配
            for i in 0..self.model.vocab_size {
                for j in 0..self.model.hidden_size {
                    let gradient = grad[i] * self.model.token_embeddings[[input_id, j]];
                    self.model.output_projection_grads[[j, i]] += gradient;
                }
            }
            
            // 埋め込み層の勾配
            for j in 0..self.model.hidden_size {
                let mut embedding_grad = 0.0;
                for i in 0..self.model.vocab_size {
                    embedding_grad += grad[i] * self.model.output_projection[[j, i]];
                }
                self.model.token_embedding_grads[[input_id, j]] += embedding_grad;
            }
            
            // 重みを更新（勾配降下法）
            self.apply_gradients();
        }
    }
    
    /// 勾配を適用して重みを更新
    fn apply_gradients(&mut self) {
        let lr = self.learning_rate;
        
        // 出力層の更新
        for i in 0..self.model.hidden_size {
            for j in 0..self.model.vocab_size {
                self.model.output_projection[[i, j]] -= lr * self.model.output_projection_grads[[i, j]];
                // 勾配をリセット
                self.model.output_projection_grads[[i, j]] = 0.0;
            }
        }
        
        // 埋め込み層の更新
        for i in 0..self.model.vocab_size {
            for j in 0..self.model.hidden_size {
                self.model.token_embeddings[[i, j]] -= lr * self.model.token_embedding_grads[[i, j]];
                // 勾配をリセット
                self.model.token_embedding_grads[[i, j]] = 0.0;
            }
        }
    }
    
    /// テキスト生成
    pub fn generate(&self, prompt: &str, max_length: usize, temperature: f32) -> String {
        // プロンプトをエンコード
        let encoded = self.tokenizer.encode(prompt);
        
        // BOS/EOSトークンを除去して実際の入力トークンのみを保持
        let mut tokens: Vec<usize> = encoded
            .into_iter()
            .filter(|&t| t != self.tokenizer.eos_token_id && t != self.tokenizer.bos_token_id)
            .collect();
        
        // プロンプトが空の場合は空文字を返す
        if tokens.is_empty() {
            return String::new();
        }
        
        // 生成されたトークンを保存
        let mut generated_tokens = Vec::new();
        
        for _ in 0..max_length {
            // コンテキストウィンドウ内に収める（BOSトークンを追加）
            let mut context = vec![self.tokenizer.bos_token_id];
            let context_start = tokens.len().saturating_sub(self.model.max_seq_len - 2);
            context.extend_from_slice(&tokens[context_start..]);
            
            // 推論
            let logits = self.model.forward(&context, false);
            if let Some(last_logit) = logits.last() {
                // 温度付きサンプリング
                let probs = softmax_with_temperature(last_logit, temperature);
                let next_token = sample_from_probs(&probs);
                
                // EOSトークンが生成されたら終了
                if next_token == self.tokenizer.eos_token_id {
                    break;
                }
                
                // BOSトークンは無視して続ける
                if next_token == self.tokenizer.bos_token_id {
                    continue;
                }
                
                tokens.push(next_token);
                generated_tokens.push(next_token);
            } else {
                break;
            }
        }
        
        // 生成された部分のみをデコード
        if !generated_tokens.is_empty() {
            self.tokenizer.decode(&generated_tokens)
        } else {
            String::new()
        }
    }
}

/// Softmax関数
fn softmax(x: &Array1<f32>) -> Array1<f32> {
    let max_val = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_vals = x.mapv(|v| (v - max_val).exp());
    let sum_exp = exp_vals.sum();
    exp_vals / sum_exp
}

/// 温度付きSoftmax
fn softmax_with_temperature(x: &Array1<f32>, temperature: f32) -> Array1<f32> {
    let scaled = x.mapv(|v| v / temperature);
    softmax(&scaled)
}

/// 確率分布からサンプリング
fn sample_from_probs(probs: &Array1<f32>) -> usize {
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    
    for (idx, &prob) in probs.iter().enumerate() {
        cumsum += prob;
        if cumsum > r {
            return idx;
        }
    }
    
    probs.len() - 1
}

/// シンプルなトークナイザー
#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    vocab: HashMap<String, usize>,
    id_to_token: Vec<String>,
    pub eos_token_id: usize,
    pub bos_token_id: usize,
    pub unk_token_id: usize,
}

impl SimpleTokenizer {
    pub fn new() -> Self {
        let mut tokenizer = SimpleTokenizer {
            vocab: HashMap::new(),
            id_to_token: Vec::new(),
            eos_token_id: 0,
            bos_token_id: 1,
            unk_token_id: 2,
        };
        
        tokenizer.vocab.insert("<eos>".to_string(), 0);
        tokenizer.vocab.insert("<bos>".to_string(), 1);
        tokenizer.vocab.insert("<unk>".to_string(), 2);
        tokenizer.id_to_token.push("<eos>".to_string());
        tokenizer.id_to_token.push("<bos>".to_string());
        tokenizer.id_to_token.push("<unk>".to_string());
        
        tokenizer
    }
    
    pub fn build_vocab(&mut self, texts: &[&str]) {
        let mut id = self.id_to_token.len();
        
        for text in texts {
            for token in text.split_whitespace() {
                if !self.vocab.contains_key(token) {
                    self.vocab.insert(token.to_string(), id);
                    self.id_to_token.push(token.to_string());
                    id += 1;
                }
            }
        }
    }
    
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = vec![];
        
        // BOSトークンを追加
        tokens.push(self.bos_token_id);
        
        // テキストをトークン化
        for token in text.split_whitespace() {
            tokens.push(*self.vocab.get(token).unwrap_or(&self.unk_token_id));
        }
        
        // EOSトークンを追加
        tokens.push(self.eos_token_id);
        tokens
    }
    
    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .filter_map(|&id| {
                let token = self.id_to_token.get(id)?;
                if matches!(token.as_str(), "<eos>" | "<bos>" | "<unk>") {
                    None
                } else {
                    Some(token.clone())
                }
            })
            .collect::<Vec<String>>()
            .join(" ")
    }
    
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}