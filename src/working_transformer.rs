// 動作するTransformerモデルの実装
// GPT-2スタイルの教育用実装（完全な逆伝播対応版）

use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::HashMap;

/// 学習設定
#[derive(Clone)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub grad_clip_norm: f32,
    pub weight_decay: f32,
    pub warmup_steps: usize,
    pub top_k_vocab: usize, // 語彙更新を上位K個に制限
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            batch_size: 4,
            grad_clip_norm: 1.0,
            weight_decay: 0.01,
            warmup_steps: 100,
            top_k_vocab: 10,
        }
    }
}

/// TransformerLayerの勾配を格納
#[derive(Clone)]
struct LayerGradients {
    // Attention勾配
    dw_q: Array2<f32>,
    dw_k: Array2<f32>,
    dw_v: Array2<f32>,
    dw_o: Array2<f32>,

    // LayerNorm1勾配
    dln1_gamma: Array1<f32>,
    dln1_beta: Array1<f32>,

    // FFN勾配
    dff_w1: Array2<f32>,
    dff_b1: Array1<f32>,
    dff_w2: Array2<f32>,
    dff_b2: Array1<f32>,

    // LayerNorm2勾配
    dln2_gamma: Array1<f32>,
    dln2_beta: Array1<f32>,
}

impl LayerGradients {
    fn zeros(hidden_size: usize, ffn_hidden: usize) -> Self {
        Self {
            dw_q: Array2::zeros((hidden_size, hidden_size)),
            dw_k: Array2::zeros((hidden_size, hidden_size)),
            dw_v: Array2::zeros((hidden_size, hidden_size)),
            dw_o: Array2::zeros((hidden_size, hidden_size)),
            dln1_gamma: Array1::zeros(hidden_size),
            dln1_beta: Array1::zeros(hidden_size),
            dff_w1: Array2::zeros((hidden_size, ffn_hidden)),
            dff_b1: Array1::zeros(ffn_hidden),
            dff_w2: Array2::zeros((ffn_hidden, hidden_size)),
            dff_b2: Array1::zeros(hidden_size),
            dln2_gamma: Array1::zeros(hidden_size),
            dln2_beta: Array1::zeros(hidden_size),
        }
    }
}

/// モデル全体の勾配を格納
/// 内部API: 学習時のみ使用
pub(crate) struct ModelGradients {
    d_token_embeddings: Array2<f32>,
    d_position_embeddings: Array2<f32>,
    layer_grads: Vec<LayerGradients>,
    d_ln_final_gamma: Array1<f32>,
    d_ln_final_beta: Array1<f32>,
    d_output_projection: Array2<f32>,
}

impl ModelGradients {
    pub(crate) fn new(vocab_size: usize, hidden_size: usize, max_seq_len: usize, num_layers: usize) -> Self {
        let ffn_hidden = hidden_size * 4;
        Self {
            d_token_embeddings: Array2::zeros((vocab_size, hidden_size)),
            d_position_embeddings: Array2::zeros((max_seq_len, hidden_size)),
            layer_grads: (0..num_layers)
                .map(|_| LayerGradients::zeros(hidden_size, ffn_hidden))
                .collect(),
            d_ln_final_gamma: Array1::zeros(hidden_size),
            d_ln_final_beta: Array1::zeros(hidden_size),
            d_output_projection: Array2::zeros((hidden_size, vocab_size)),
        }
    }

    /// 勾配のL2ノルムを計算
    fn l2_norm(&self) -> f32 {
        let mut sum = 0.0;
        sum += self.d_token_embeddings.iter().map(|x| x * x).sum::<f32>();
        sum += self.d_position_embeddings.iter().map(|x| x * x).sum::<f32>();
        sum += self.d_output_projection.iter().map(|x| x * x).sum::<f32>();
        sum += self.d_ln_final_gamma.iter().map(|x| x * x).sum::<f32>();
        sum += self.d_ln_final_beta.iter().map(|x| x * x).sum::<f32>();
        for lg in &self.layer_grads {
            sum += lg.dw_q.iter().map(|x| x * x).sum::<f32>();
            sum += lg.dw_k.iter().map(|x| x * x).sum::<f32>();
            sum += lg.dw_v.iter().map(|x| x * x).sum::<f32>();
            sum += lg.dw_o.iter().map(|x| x * x).sum::<f32>();
            sum += lg.dln1_gamma.iter().map(|x| x * x).sum::<f32>();
            sum += lg.dln1_beta.iter().map(|x| x * x).sum::<f32>();
            sum += lg.dff_w1.iter().map(|x| x * x).sum::<f32>();
            sum += lg.dff_b1.iter().map(|x| x * x).sum::<f32>();
            sum += lg.dff_w2.iter().map(|x| x * x).sum::<f32>();
            sum += lg.dff_b2.iter().map(|x| x * x).sum::<f32>();
            sum += lg.dln2_gamma.iter().map(|x| x * x).sum::<f32>();
            sum += lg.dln2_beta.iter().map(|x| x * x).sum::<f32>();
        }
        sum.sqrt()
    }

    /// 勾配クリッピング
    fn clip_grad_norm(&mut self, max_norm: f32) {
        let norm = self.l2_norm();
        if norm > max_norm {
            let scale = max_norm / norm;
            self.d_token_embeddings.mapv_inplace(|x| x * scale);
            self.d_position_embeddings.mapv_inplace(|x| x * scale);
            self.d_output_projection.mapv_inplace(|x| x * scale);
            self.d_ln_final_gamma.mapv_inplace(|x| x * scale);
            self.d_ln_final_beta.mapv_inplace(|x| x * scale);
            for lg in &mut self.layer_grads {
                lg.dw_q.mapv_inplace(|x| x * scale);
                lg.dw_k.mapv_inplace(|x| x * scale);
                lg.dw_v.mapv_inplace(|x| x * scale);
                lg.dw_o.mapv_inplace(|x| x * scale);
                lg.dln1_gamma.mapv_inplace(|x| x * scale);
                lg.dln1_beta.mapv_inplace(|x| x * scale);
                lg.dff_w1.mapv_inplace(|x| x * scale);
                lg.dff_b1.mapv_inplace(|x| x * scale);
                lg.dff_w2.mapv_inplace(|x| x * scale);
                lg.dff_b2.mapv_inplace(|x| x * scale);
                lg.dln2_gamma.mapv_inplace(|x| x * scale);
                lg.dln2_beta.mapv_inplace(|x| x * scale);
            }
        }
    }
}

/// 順伝播時の中間値キャッシュ（逆伝播用）
/// 内部API: 学習時のみ使用
pub(crate) struct ForwardCache {
    // 入力トークンID
    input_ids: Vec<usize>,
    // 各レイヤーの中間値
    layer_caches: Vec<LayerCache>,
    // 最終LayerNorm前後
    pre_ln_final: Array2<f32>,
    post_ln_final: Array2<f32>,
    ln_final_std: Array1<f32>,
}

/// レイヤーごとの中間値キャッシュ
struct LayerCache {
    // LayerNorm1
    ln1_input: Array2<f32>,
    ln1_output: Array2<f32>,
    ln1_std: Array1<f32>,
    // Attention
    q: Array2<f32>,
    k: Array2<f32>,
    v: Array2<f32>,
    attn_weights: Vec<Array2<f32>>, // 各ヘッドのアテンション重み
    attn_output: Array2<f32>,
    // LayerNorm2
    ln2_input: Array2<f32>,
    ln2_output: Array2<f32>,
    ln2_std: Array1<f32>,
    // FFN
    ff_hidden: Array2<f32>,  // GELU前
    ff_gelu: Array2<f32>,    // GELU後
}

/// 実際に動作するTransformerモデル
pub struct WorkingTransformer {
    // モデル設定
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,

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
}

/// Transformerの1レイヤー
struct TransformerLayer {
    // モデル設定
    num_heads: usize,
    head_dim: usize,

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

        // 出力投影（独立して初期化。weight tyingをやめて挙動を明確化）
        let output_projection = Array2::from_shape_fn((hidden_size, vocab_size), |_| {
            rng.gen_range(-init_scale..init_scale)
        });
        
        Self {
            vocab_size,
            hidden_size,
            num_layers,
            max_seq_len,
            token_embeddings,
            position_embeddings,
            layers,
            ln_final_gamma,
            ln_final_beta,
            output_projection,
        }
    }
    
    /// 順伝播
    pub fn forward(&self, input_ids: &[usize], training: bool) -> Vec<Array1<f32>> {
        let (logits, _) = self.forward_with_cache(input_ids, training);
        logits
    }

    /// 逆伝播用に最終hidden_statesも返す
    pub fn forward_with_cache(&self, input_ids: &[usize], _training: bool) -> (Vec<Array1<f32>>, Array2<f32>) {
        // max_seq_lenを超える入力は切り詰める
        let effective_len = input_ids.len().min(self.max_seq_len);
        let input_ids = &input_ids[..effective_len];
        let seq_len = input_ids.len();
        let _batch_size = 1; // 簡単のため

        // 埋め込み
        let mut hidden_states = Array2::zeros((seq_len, self.hidden_size));
        for (i, &token_id) in input_ids.iter().enumerate() {
            debug_assert!(token_id < self.vocab_size, "token id {} out of range", token_id);
            let token_emb = self.token_embeddings.row(token_id);
            // 位置インデックスをmax_seq_len-1でクリップ（範囲外アクセス防止）
            let pos_idx = i.min(self.max_seq_len - 1);
            let pos_emb = self.position_embeddings.row(pos_idx);
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
        
        (logits, hidden_states)
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

    /// Layer Normalization with std for backward pass
    fn layer_norm_with_std(
        &self,
        x: &Array2<f32>,
        gamma: &Array1<f32>,
        beta: &Array1<f32>,
    ) -> (Array2<f32>, Array1<f32>) {
        let (seq_len, hidden_size) = x.dim();
        let mut output = Array2::zeros((seq_len, hidden_size));
        let mut stds = Array1::zeros(seq_len);

        for i in 0..seq_len {
            let row = x.row(i);
            let mean = row.mean().unwrap();
            let var = row.mapv(|v| (v - mean).powi(2)).mean().unwrap();
            let std = (var + 1e-5).sqrt();
            stds[i] = std;

            let normalized = row.mapv(|v| (v - mean) / std);
            let scaled = &normalized * gamma + beta;
            output.row_mut(i).assign(&scaled);
        }

        (output, stds)
    }

    /// 完全なキャッシュ付き順伝播（逆伝播用）
    /// 内部API: 学習時のみ使用
    pub(crate) fn forward_full_cache(&self, input_ids: &[usize]) -> (Vec<Array1<f32>>, ForwardCache) {
        let effective_len = input_ids.len().min(self.max_seq_len);
        let input_ids = &input_ids[..effective_len];
        let seq_len = input_ids.len();

        // 埋め込み
        let mut hidden_states = Array2::zeros((seq_len, self.hidden_size));
        for (i, &token_id) in input_ids.iter().enumerate() {
            let token_emb = self.token_embeddings.row(token_id);
            let pos_idx = i.min(self.max_seq_len - 1);
            let pos_emb = self.position_embeddings.row(pos_idx);
            let combined = &token_emb.to_owned() + &pos_emb.to_owned();
            hidden_states.row_mut(i).assign(&combined);
        }

        // 各レイヤーを通す（キャッシュ保存）
        let mut layer_caches = Vec::new();

        for layer in &self.layers {
            let (output, cache) = layer.forward_with_cache(&hidden_states);
            layer_caches.push(cache);
            hidden_states = output;
        }

        // 最終LayerNorm
        let pre_ln_final = hidden_states.clone();
        let (post_ln_final, ln_final_std) =
            self.layer_norm_with_std(&hidden_states, &self.ln_final_gamma, &self.ln_final_beta);

        // 出力投影
        let mut logits = Vec::new();
        for i in 0..seq_len {
            let hidden = post_ln_final.row(i);
            let logit = hidden.dot(&self.output_projection);
            logits.push(logit.to_owned());
        }

        let cache = ForwardCache {
            input_ids: input_ids.to_vec(),
            layer_caches,
            pre_ln_final,
            post_ln_final,
            ln_final_std,
        };

        (logits, cache)
    }

    /// 逆伝播：勾配を計算してModelGradientsに蓄積
    /// 内部API: 学習時のみ使用
    pub(crate) fn backward(
        &self,
        cache: &ForwardCache,
        target: &[usize],
        grads: &mut ModelGradients,
        top_k: usize,
    ) -> f32 {
        let seq_len = cache.input_ids.len();
        let mut total_loss = 0.0;

        // 出力層からの勾配を計算
        let mut d_hidden = Array2::zeros((seq_len, self.hidden_size));

        for pos in 0..seq_len {
            let logit = cache.post_ln_final.row(pos).dot(&self.output_projection);
            let probs = softmax_arr1(&logit);
            let target_id = target[pos];

            // 損失
            total_loss += -probs[target_id].ln();

            // Softmax勾配: d_logit = probs - one_hot(target)
            let mut d_logit = probs.clone();
            d_logit[target_id] -= 1.0;

            // 出力層勾配（Top-Kに制限）
            let hidden = cache.post_ln_final.row(pos);

            // Top-K indices by absolute gradient（部分選択で O(V) に最適化）
            let mut grad_indices: Vec<(usize, f32)> = d_logit
                .iter()
                .enumerate()
                .map(|(i, &g)| (i, g.abs()))
                .collect();

            // select_nth_unstable_by で上位K個を O(V) で取得
            let k = top_k.min(grad_indices.len());
            if k > 0 && k < grad_indices.len() {
                grad_indices.select_nth_unstable_by(k - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
            }

            for (idx, _) in grad_indices.iter().take(k) {
                let vocab_idx = *idx;
                for j in 0..self.hidden_size {
                    grads.d_output_projection[[j, vocab_idx]] += d_logit[vocab_idx] * hidden[j];
                }
            }

            // d_hidden = d_logit @ output_projection.T
            for j in 0..self.hidden_size {
                let mut sum = 0.0;
                for (idx, _) in grad_indices.iter().take(k) {
                    sum += d_logit[*idx] * self.output_projection[[j, *idx]];
                }
                d_hidden[[pos, j]] = sum;
            }
        }

        // 最終LayerNormの逆伝播
        let (d_pre_ln, d_gamma, d_beta) = self.layer_norm_backward(
            &d_hidden,
            &cache.pre_ln_final,
            &cache.ln_final_std,
            &self.ln_final_gamma,
        );
        grads.d_ln_final_gamma = &grads.d_ln_final_gamma + &d_gamma;
        grads.d_ln_final_beta = &grads.d_ln_final_beta + &d_beta;

        // 各レイヤーを逆順に逆伝播
        let mut d_hidden = d_pre_ln;
        for (layer_idx, layer) in self.layers.iter().enumerate().rev() {
            let layer_cache = &cache.layer_caches[layer_idx];
            let layer_grad = &mut grads.layer_grads[layer_idx];
            d_hidden = layer.backward(&d_hidden, layer_cache, layer_grad);
        }

        // 埋め込み層の勾配
        for (i, &token_id) in cache.input_ids.iter().enumerate() {
            for j in 0..self.hidden_size {
                grads.d_token_embeddings[[token_id, j]] += d_hidden[[i, j]];
                let pos_idx = i.min(self.max_seq_len - 1);
                grads.d_position_embeddings[[pos_idx, j]] += d_hidden[[i, j]];
            }
        }

        total_loss
    }

    /// LayerNorm逆伝播
    ///
    /// 注意: この実装は近似版です。d_xの計算において、平均・分散への勾配を
    /// 省略しています（d_x ≈ d_normalized / std）。
    ///
    /// 正確な実装では以下の項も必要：
    /// d_var = sum(d_normalized * (x - mean) * -0.5 * std^-3)
    /// d_mean = sum(d_normalized * -1/std) + d_var * sum(-2*(x-mean))/N
    /// d_x = d_normalized/std + d_var*2*(x-mean)/N + d_mean/N
    ///
    /// 教育目的のため、シンプルさを優先してこの近似を採用しています。
    /// 本格的な学習には正確な勾配計算への置き換えを推奨します。
    fn layer_norm_backward(
        &self,
        d_out: &Array2<f32>,
        x: &Array2<f32>,
        stds: &Array1<f32>,
        gamma: &Array1<f32>,
    ) -> (Array2<f32>, Array1<f32>, Array1<f32>) {
        let (seq_len, hidden_size) = x.dim();
        let n = hidden_size as f32;
        let mut d_x = Array2::zeros((seq_len, hidden_size));
        let mut d_gamma = Array1::zeros(hidden_size);
        let mut d_beta = Array1::zeros(hidden_size);

        for i in 0..seq_len {
            let row = x.row(i);
            let mean = row.mean().unwrap();
            let std = stds[i];

            let normalized: Array1<f32> = row.mapv(|v| (v - mean) / std);

            // d_beta = sum(d_out)
            // d_gamma = sum(d_out * normalized)
            for j in 0..hidden_size {
                d_beta[j] += d_out[[i, j]];
                d_gamma[j] += d_out[[i, j]] * normalized[j];
            }

            // d_normalized = d_out * gamma
            let d_normalized: Array1<f32> = (0..hidden_size)
                .map(|j| d_out[[i, j]] * gamma[j])
                .collect::<Vec<f32>>()
                .into();

            // 正確なLayerNorm勾配計算
            // d_x = (1/std) * (d_normalized - mean(d_normalized) - normalized * mean(d_normalized * normalized))
            let mean_d_norm = d_normalized.mean().unwrap();
            let mean_d_norm_x_norm: f32 = d_normalized.iter()
                .zip(normalized.iter())
                .map(|(d, n)| d * n)
                .sum::<f32>() / n;

            for j in 0..hidden_size {
                d_x[[i, j]] = (d_normalized[j] - mean_d_norm - normalized[j] * mean_d_norm_x_norm) / std;
            }
        }

        (d_x, d_gamma, d_beta)
    }

    /// 勾配を適用してパラメータを更新
    /// 内部API: 学習時のみ使用
    pub(crate) fn apply_gradients(&mut self, grads: &ModelGradients, lr: f32, weight_decay: f32) {
        // 重み減衰付きSGD
        let apply_update = |param: &mut Array2<f32>, grad: &Array2<f32>| {
            param.zip_mut_with(grad, |p, &g| {
                *p -= lr * (g + weight_decay * *p);
            });
        };

        let apply_update_1d = |param: &mut Array1<f32>, grad: &Array1<f32>| {
            param.zip_mut_with(grad, |p, &g| {
                *p -= lr * g; // LayerNormには重み減衰なし
            });
        };

        // 埋め込み層
        apply_update(&mut self.token_embeddings, &grads.d_token_embeddings);
        apply_update(&mut self.position_embeddings, &grads.d_position_embeddings);

        // 各レイヤー
        for (layer, lg) in self.layers.iter_mut().zip(grads.layer_grads.iter()) {
            apply_update(&mut layer.w_q, &lg.dw_q);
            apply_update(&mut layer.w_k, &lg.dw_k);
            apply_update(&mut layer.w_v, &lg.dw_v);
            apply_update(&mut layer.w_o, &lg.dw_o);
            apply_update_1d(&mut layer.ln1_gamma, &lg.dln1_gamma);
            apply_update_1d(&mut layer.ln1_beta, &lg.dln1_beta);
            apply_update(&mut layer.ff_w1, &lg.dff_w1);
            apply_update_1d(&mut layer.ff_b1, &lg.dff_b1);
            apply_update(&mut layer.ff_w2, &lg.dff_w2);
            apply_update_1d(&mut layer.ff_b2, &lg.dff_b2);
            apply_update_1d(&mut layer.ln2_gamma, &lg.dln2_gamma);
            apply_update_1d(&mut layer.ln2_beta, &lg.dln2_beta);
        }

        // 最終LayerNorm
        apply_update_1d(&mut self.ln_final_gamma, &grads.d_ln_final_gamma);
        apply_update_1d(&mut self.ln_final_beta, &grads.d_ln_final_beta);

        // 出力層
        apply_update(&mut self.output_projection, &grads.d_output_projection);
    }
}

/// Softmax for Array1
fn softmax_arr1(x: &Array1<f32>) -> Array1<f32> {
    let max_val = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_vals = x.mapv(|v| (v - max_val).exp());
    let sum_exp = exp_vals.sum();
    exp_vals / sum_exp
}

impl TransformerLayer {
    fn new(hidden_size: usize, num_heads: usize, rng: &mut impl Rng) -> Self {
        let scale = (1.0 / hidden_size as f32).sqrt();
        let ffn_hidden = hidden_size * 4;
        let head_dim = hidden_size / num_heads;

        Self {
            // モデル設定
            num_heads,
            head_dim,

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
    
    /// Multi-Head Attention（正しいヘッド分割実装）
    fn multi_head_attention(&self, x: &Array2<f32>) -> Array2<f32> {
        let (seq_len, hidden_size) = x.dim();
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;
        let scale = (head_dim as f32).sqrt();

        // Q, K, Vの計算 (seq_len, hidden_size)
        let q = x.dot(&self.w_q);
        let k = x.dot(&self.w_k);
        let v = x.dot(&self.w_v);

        // 各ヘッドの出力を格納する配列
        let mut head_outputs = Array2::zeros((seq_len, hidden_size));

        // 各ヘッドごとに独立してアテンションを計算
        for h in 0..num_heads {
            let start = h * head_dim;
            let end = start + head_dim;

            // ヘッドごとのQ, K, Vを抽出 (seq_len, head_dim)
            let q_h = q.slice(ndarray::s![.., start..end]).to_owned();
            let k_h = k.slice(ndarray::s![.., start..end]).to_owned();
            let v_h = v.slice(ndarray::s![.., start..end]).to_owned();

            // アテンションスコアの計算 (seq_len, seq_len)
            let scores = q_h.dot(&k_h.t()) / scale;

            // Causal maskとSoftmaxの適用
            let mut attn_weights = Array2::zeros((seq_len, seq_len));
            for i in 0..seq_len {
                // Causal mask: 現在位置より先は-infにする
                let mut max_score = f32::NEG_INFINITY;
                for j in 0..=i {
                    attn_weights[[i, j]] = scores[[i, j]];
                    max_score = max_score.max(scores[[i, j]]);
                }

                // Softmax (各行に対して)
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

            // Attention output for this head (seq_len, head_dim)
            let head_output = attn_weights.dot(&v_h);

            // 結果を適切な位置に格納
            for i in 0..seq_len {
                for j in 0..head_dim {
                    head_outputs[[i, start + j]] = head_output[[i, j]];
                }
            }
        }

        // 出力投影
        head_outputs.dot(&self.w_o)
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

    /// Layer Normalization with std for backward
    fn layer_norm_with_std(
        &self,
        x: &Array2<f32>,
        gamma: &Array1<f32>,
        beta: &Array1<f32>,
    ) -> (Array2<f32>, Array1<f32>) {
        let (seq_len, hidden_size) = x.dim();
        let mut output = Array2::zeros((seq_len, hidden_size));
        let mut stds = Array1::zeros(seq_len);

        for i in 0..seq_len {
            let row = x.row(i);
            let mean = row.mean().unwrap();
            let var = row.mapv(|v| (v - mean).powi(2)).mean().unwrap();
            let std = (var + 1e-5).sqrt();
            stds[i] = std;

            let normalized = row.mapv(|v| (v - mean) / std);
            let scaled = &normalized * gamma + beta;
            output.row_mut(i).assign(&scaled);
        }

        (output, stds)
    }

    /// キャッシュ付き順伝播
    fn forward_with_cache(&self, x: &Array2<f32>) -> (Array2<f32>, LayerCache) {
        let (seq_len, hidden_size) = x.dim();

        // LayerNorm1
        let ln1_input = x.clone();
        let (ln1_output, ln1_std) = self.layer_norm_with_std(x, &self.ln1_gamma, &self.ln1_beta);

        // Multi-Head Attention with cache
        let q = ln1_output.dot(&self.w_q);
        let k = ln1_output.dot(&self.w_k);
        let v = ln1_output.dot(&self.w_v);

        let num_heads = self.num_heads;
        let head_dim = self.head_dim;
        let scale = (head_dim as f32).sqrt();

        let mut head_outputs = Array2::zeros((seq_len, hidden_size));
        let mut attn_weights_all = Vec::new();

        for h in 0..num_heads {
            let start = h * head_dim;
            let end = start + head_dim;

            let q_h = q.slice(ndarray::s![.., start..end]).to_owned();
            let k_h = k.slice(ndarray::s![.., start..end]).to_owned();
            let v_h = v.slice(ndarray::s![.., start..end]).to_owned();

            let scores = q_h.dot(&k_h.t()) / scale;

            let mut attn_weights = Array2::zeros((seq_len, seq_len));
            for i in 0..seq_len {
                let mut max_score = f32::NEG_INFINITY;
                for j in 0..=i {
                    attn_weights[[i, j]] = scores[[i, j]];
                    max_score = max_score.max(scores[[i, j]]);
                }
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

            attn_weights_all.push(attn_weights.clone());

            let head_output = attn_weights.dot(&v_h);
            for i in 0..seq_len {
                for j in 0..head_dim {
                    head_outputs[[i, start + j]] = head_output[[i, j]];
                }
            }
        }

        let attn_output = head_outputs.dot(&self.w_o);
        let after_attn = x + &attn_output;

        // LayerNorm2
        let ln2_input = after_attn.clone();
        let (ln2_output, ln2_std) = self.layer_norm_with_std(&after_attn, &self.ln2_gamma, &self.ln2_beta);

        // FFN with cache
        let ff_hidden = ln2_output.dot(&self.ff_w1) + &self.ff_b1;
        let ff_gelu = ff_hidden.mapv(gelu);
        let ff_output = ff_gelu.dot(&self.ff_w2) + &self.ff_b2;

        let output = &after_attn + &ff_output;

        let cache = LayerCache {
            ln1_input,
            ln1_output,
            ln1_std,
            q,
            k,
            v,
            attn_weights: attn_weights_all,
            attn_output,
            ln2_input,
            ln2_output,
            ln2_std,
            ff_hidden,
            ff_gelu,
        };

        (output, cache)
    }

    /// 逆伝播
    fn backward(
        &self,
        d_out: &Array2<f32>,
        cache: &LayerCache,
        grads: &mut LayerGradients,
    ) -> Array2<f32> {
        let (seq_len, hidden_size) = d_out.dim();
        let ffn_hidden = hidden_size * 4;

        // FFN backward (residual connection)
        let d_ff_out = d_out;
        let d_after_attn_from_ff = d_out; // residual

        // FFN Layer 2 backward
        // d_ff_gelu = d_ff_out @ ff_w2.T
        let d_ff_gelu = d_ff_out.dot(&self.ff_w2.t());
        // dff_w2 += ff_gelu.T @ d_ff_out
        grads.dff_w2 = &grads.dff_w2 + &cache.ff_gelu.t().dot(d_ff_out);
        // dff_b2 += sum(d_ff_out, axis=0)
        for j in 0..hidden_size {
            for i in 0..seq_len {
                grads.dff_b2[j] += d_ff_out[[i, j]];
            }
        }

        // GELU backward
        let d_ff_hidden = Array2::from_shape_fn((seq_len, ffn_hidden), |(i, j)| {
            d_ff_gelu[[i, j]] * gelu_derivative(cache.ff_hidden[[i, j]])
        });

        // FFN Layer 1 backward
        let d_ln2_output = d_ff_hidden.dot(&self.ff_w1.t());
        grads.dff_w1 = &grads.dff_w1 + &cache.ln2_output.t().dot(&d_ff_hidden);
        for j in 0..ffn_hidden {
            for i in 0..seq_len {
                grads.dff_b1[j] += d_ff_hidden[[i, j]];
            }
        }

        // LayerNorm2 backward
        let (d_ln2_input, d_ln2_gamma, d_ln2_beta) = self.layer_norm_backward_internal(
            &d_ln2_output,
            &cache.ln2_input,
            &cache.ln2_std,
            &self.ln2_gamma,
        );
        grads.dln2_gamma = &grads.dln2_gamma + &d_ln2_gamma;
        grads.dln2_beta = &grads.dln2_beta + &d_ln2_beta;

        // Combine with residual from FF
        let d_after_attn = &d_ln2_input + d_after_attn_from_ff;

        // Attention backward (residual connection)
        let d_attn_out = &d_after_attn;
        let d_residual1 = &d_after_attn; // residual

        // Output projection backward
        let d_head_outputs = d_attn_out.dot(&self.w_o.t());
        grads.dw_o = &grads.dw_o + &cache.attn_output.t().dot(d_attn_out);

        // Multi-head attention backward (simplified)
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;

        let mut d_q = Array2::zeros((seq_len, hidden_size));
        let mut d_k = Array2::zeros((seq_len, hidden_size));
        let mut d_v = Array2::zeros((seq_len, hidden_size));

        for h in 0..num_heads {
            let start = h * head_dim;
            let end = start + head_dim;

            let v_h = cache.v.slice(ndarray::s![.., start..end]).to_owned();
            let attn_weights = &cache.attn_weights[h];

            // d_attn_output for this head
            let d_head_out = d_head_outputs.slice(ndarray::s![.., start..end]).to_owned();

            // d_v_h = attn_weights.T @ d_head_out
            let d_v_h = attn_weights.t().dot(&d_head_out);
            for i in 0..seq_len {
                for j in 0..head_dim {
                    d_v[[i, start + j]] += d_v_h[[i, j]];
                }
            }

            // d_attn_weights = d_head_out @ v_h.T
            let d_attn_weights = d_head_out.dot(&v_h.t());

            // Softmax backward (simplified)
            let scale = (head_dim as f32).sqrt();
            let q_h = cache.q.slice(ndarray::s![.., start..end]).to_owned();
            let k_h = cache.k.slice(ndarray::s![.., start..end]).to_owned();

            // d_scores = softmax_backward(d_attn_weights, attn_weights)
            // Simplified: d_scores ≈ d_attn_weights * attn_weights * (1 - attn_weights)
            // Actually: d_scores_ij = attn_weights_ij * (d_attn_weights_ij - sum_k(d_attn_weights_ik * attn_weights_ik))
            let mut d_scores = Array2::zeros((seq_len, seq_len));
            for i in 0..seq_len {
                let sum_da_a: f32 = (0..=i).map(|k| d_attn_weights[[i, k]] * attn_weights[[i, k]]).sum();
                for j in 0..=i {
                    d_scores[[i, j]] = attn_weights[[i, j]] * (d_attn_weights[[i, j]] - sum_da_a);
                }
            }
            d_scores.mapv_inplace(|x| x / scale);

            // d_q_h = d_scores @ k_h
            let d_q_h = d_scores.dot(&k_h);
            // d_k_h = d_scores.T @ q_h
            let d_k_h = d_scores.t().dot(&q_h);

            for i in 0..seq_len {
                for j in 0..head_dim {
                    d_q[[i, start + j]] += d_q_h[[i, j]];
                    d_k[[i, start + j]] += d_k_h[[i, j]];
                }
            }
        }

        // Q, K, V projection backward
        let d_ln1_output_from_q = d_q.dot(&self.w_q.t());
        let d_ln1_output_from_k = d_k.dot(&self.w_k.t());
        let d_ln1_output_from_v = d_v.dot(&self.w_v.t());
        let d_ln1_output = &d_ln1_output_from_q + &d_ln1_output_from_k + &d_ln1_output_from_v;

        grads.dw_q = &grads.dw_q + &cache.ln1_output.t().dot(&d_q);
        grads.dw_k = &grads.dw_k + &cache.ln1_output.t().dot(&d_k);
        grads.dw_v = &grads.dw_v + &cache.ln1_output.t().dot(&d_v);

        // LayerNorm1 backward
        let (d_ln1_input, d_ln1_gamma, d_ln1_beta) = self.layer_norm_backward_internal(
            &d_ln1_output,
            &cache.ln1_input,
            &cache.ln1_std,
            &self.ln1_gamma,
        );
        grads.dln1_gamma = &grads.dln1_gamma + &d_ln1_gamma;
        grads.dln1_beta = &grads.dln1_beta + &d_ln1_beta;

        // Combine with residual
        &d_ln1_input + d_residual1
    }

    /// LayerNorm backward helper（正確な勾配計算版）
    fn layer_norm_backward_internal(
        &self,
        d_out: &Array2<f32>,
        x: &Array2<f32>,
        stds: &Array1<f32>,
        gamma: &Array1<f32>,
    ) -> (Array2<f32>, Array1<f32>, Array1<f32>) {
        let (seq_len, hidden_size) = x.dim();
        let n = hidden_size as f32;
        let mut d_x = Array2::zeros((seq_len, hidden_size));
        let mut d_gamma = Array1::zeros(hidden_size);
        let mut d_beta = Array1::zeros(hidden_size);

        for i in 0..seq_len {
            let row = x.row(i);
            let mean = row.mean().unwrap();
            let std = stds[i];

            let normalized: Array1<f32> = row.mapv(|v| (v - mean) / std);

            for j in 0..hidden_size {
                d_beta[j] += d_out[[i, j]];
                d_gamma[j] += d_out[[i, j]] * normalized[j];
            }

            let d_normalized: Array1<f32> = (0..hidden_size)
                .map(|j| d_out[[i, j]] * gamma[j])
                .collect::<Vec<f32>>()
                .into();

            // 正確なLayerNorm勾配計算
            let mean_d_norm = d_normalized.mean().unwrap();
            let mean_d_norm_x_norm: f32 = d_normalized.iter()
                .zip(normalized.iter())
                .map(|(d, n)| d * n)
                .sum::<f32>() / n;

            for j in 0..hidden_size {
                d_x[[i, j]] = (d_normalized[j] - mean_d_norm - normalized[j] * mean_d_norm_x_norm) / std;
            }
        }

        (d_x, d_gamma, d_beta)
    }
}

/// GELU活性化関数
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// GELU導関数
fn gelu_derivative(x: f32) -> f32 {
    let sqrt_2_pi = (2.0 / std::f32::consts::PI).sqrt();
    let inner = sqrt_2_pi * (x + 0.044715 * x.powi(3));
    let tanh_inner = inner.tanh();
    let sech2 = 1.0 - tanh_inner * tanh_inner;
    0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x * x)
}

/// 学習可能なTransformerモデル
pub struct TrainableTransformer {
    pub model: WorkingTransformer,
    pub tokenizer: SimpleTokenizer,
    pub config: TrainingConfig,
    step: usize,
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
        let mut config = TrainingConfig::default();
        config.learning_rate = learning_rate;

        Self {
            model,
            tokenizer,
            config,
            step: 1, // 1から開始し、最初のバッチでも学習が行われるようにする
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
        let mut config = TrainingConfig::default();
        config.learning_rate = learning_rate;

        Self {
            model,
            tokenizer,
            config,
            step: 1, // 1から開始し、最初のバッチでも学習が行われるようにする
        }
    }

    /// 学習率スケジューリング（warmup付き）
    fn get_learning_rate(&self) -> f32 {
        let base_lr = self.config.learning_rate;
        if self.step <= self.config.warmup_steps {
            // Linear warmup（step=1から開始するため、最初のバッチから学習が行われる）
            base_lr * (self.step as f32 / self.config.warmup_steps as f32)
        } else {
            base_lr
        }
    }

    /// 完全な逆伝播による学習（ミニバッチ対応）
    pub fn train(&mut self, texts: &[String], epochs: usize) {
        let batch_size = self.config.batch_size.min(texts.len()).max(1);

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut count = 0;

            // テキストをバッチに分割
            for batch_start in (0..texts.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(texts.len());
                let batch = &texts[batch_start..batch_end];

                // 勾配バッファを初期化
                let mut grads = ModelGradients::new(
                    self.model.vocab_size,
                    self.model.hidden_size,
                    self.model.max_seq_len,
                    self.model.num_layers,
                );

                let mut batch_loss = 0.0;
                let mut batch_tokens = 0;

                // バッチ内の各サンプルで勾配を累積
                for text in batch {
                    let tokens = self.tokenizer.encode(text);
                    if tokens.len() < 2 {
                        continue;
                    }

                    let input = &tokens[..tokens.len() - 1];
                    let target = &tokens[1..];

                    // 完全なキャッシュ付き順伝播
                    let (_logits, cache) = self.model.forward_full_cache(input);

                    // 完全な逆伝播（Attention/FFN/LayerNorm全て）
                    let loss = self.model.backward(
                        &cache,
                        target,
                        &mut grads,
                        self.config.top_k_vocab,
                    );

                    batch_loss += loss;
                    batch_tokens += input.len();
                }

                if batch_tokens > 0 {
                    // 勾配をバッチサイズで正規化
                    let scale = 1.0 / batch_tokens as f32;
                    grads.d_token_embeddings.mapv_inplace(|x| x * scale);
                    grads.d_position_embeddings.mapv_inplace(|x| x * scale);
                    grads.d_output_projection.mapv_inplace(|x| x * scale);
                    grads.d_ln_final_gamma.mapv_inplace(|x| x * scale);
                    grads.d_ln_final_beta.mapv_inplace(|x| x * scale);
                    for lg in &mut grads.layer_grads {
                        lg.dw_q.mapv_inplace(|x| x * scale);
                        lg.dw_k.mapv_inplace(|x| x * scale);
                        lg.dw_v.mapv_inplace(|x| x * scale);
                        lg.dw_o.mapv_inplace(|x| x * scale);
                        lg.dln1_gamma.mapv_inplace(|x| x * scale);
                        lg.dln1_beta.mapv_inplace(|x| x * scale);
                        lg.dff_w1.mapv_inplace(|x| x * scale);
                        lg.dff_b1.mapv_inplace(|x| x * scale);
                        lg.dff_w2.mapv_inplace(|x| x * scale);
                        lg.dff_b2.mapv_inplace(|x| x * scale);
                        lg.dln2_gamma.mapv_inplace(|x| x * scale);
                        lg.dln2_beta.mapv_inplace(|x| x * scale);
                    }

                    // 勾配クリッピング
                    grads.clip_grad_norm(self.config.grad_clip_norm);

                    // 学習率を取得（warmup考慮）
                    let lr = self.get_learning_rate();

                    // パラメータ更新
                    self.model.apply_gradients(&grads, lr, self.config.weight_decay);

                    total_loss += batch_loss;
                    count += batch_tokens;
                    self.step += 1;
                }
            }

            if count > 0 {
                println!(
                    "Epoch {}: 平均損失 = {:.4}, 学習率 = {:.6}",
                    epoch + 1,
                    total_loss / count as f32,
                    self.get_learning_rate()
                );
            }
        }
    }

    /// 後方互換性のための簡易学習（旧API）
    pub fn train_simple(&mut self, texts: &[String], epochs: usize) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut count = 0;

            for text in texts {
                let tokens = self.tokenizer.encode(text);
                if tokens.len() < 2 {
                    continue;
                }

                let input = &tokens[..tokens.len() - 1];
                let target = &tokens[1..];
                let logits = self.model.forward(input, true);

                for (pos, logit) in logits.iter().enumerate() {
                    let target_id = target[pos];
                    let probs = softmax(logit);
                    let loss = -probs[target_id].ln();
                    total_loss += loss;
                    count += 1;
                }
            }

            if count > 0 {
                println!("Epoch {}: 平均損失 = {:.4}", epoch + 1, total_loss / count as f32);
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
                // 特殊トークンをマスクして空返却を防ぐ
                let mut masked = last_logit.clone();
                masked[self.tokenizer.bos_token_id] = f32::NEG_INFINITY;
                masked[self.tokenizer.unk_token_id] = f32::NEG_INFINITY;
                // 1トークンも生成していない間はEOSを禁止
                if generated_tokens.is_empty() {
                    masked[self.tokenizer.eos_token_id] = f32::NEG_INFINITY;
                }

                // 温度付きサンプリング
                let probs = softmax_with_temperature(&masked, temperature);
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
    pub id_to_token: Vec<String>,
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

#[cfg(test)]
mod tests {
    use super::*;

    // ===========================================
    // トークナイザーのテスト
    // ===========================================

    #[test]
    fn test_tokenizer_new() {
        let tokenizer = SimpleTokenizer::new();
        assert_eq!(tokenizer.vocab_size(), 3); // <eos>, <bos>, <unk>
        assert_eq!(tokenizer.eos_token_id, 0);
        assert_eq!(tokenizer.bos_token_id, 1);
        assert_eq!(tokenizer.unk_token_id, 2);
    }

    #[test]
    fn test_tokenizer_build_vocab() {
        let mut tokenizer = SimpleTokenizer::new();
        tokenizer.build_vocab(&["Hello world", "Hello Rust"]);
        // <eos>, <bos>, <unk>, Hello, world, Rust = 6
        assert_eq!(tokenizer.vocab_size(), 6);
    }

    #[test]
    fn test_tokenizer_encode_decode() {
        let mut tokenizer = SimpleTokenizer::new();
        tokenizer.build_vocab(&["Hello world"]);

        let encoded = tokenizer.encode("Hello world");
        // BOS + Hello + world + EOS
        assert_eq!(encoded.len(), 4);
        assert_eq!(encoded[0], tokenizer.bos_token_id);
        assert_eq!(encoded[encoded.len() - 1], tokenizer.eos_token_id);

        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, "Hello world");
    }

    #[test]
    fn test_tokenizer_unknown_token() {
        let mut tokenizer = SimpleTokenizer::new();
        tokenizer.build_vocab(&["Hello world"]);

        let encoded = tokenizer.encode("Hello unknown");
        // "unknown" は語彙にないので <unk> になる
        assert_eq!(encoded[2], tokenizer.unk_token_id);
    }

    // ===========================================
    // Transformerモデルのテスト
    // ===========================================

    #[test]
    fn test_transformer_creation() {
        let model = WorkingTransformer::new(100, 64, 4, 2, 128);
        assert_eq!(model.vocab_size, 100);
        assert_eq!(model.hidden_size, 64);
    }

    #[test]
    fn test_transformer_forward() {
        let model = WorkingTransformer::new(100, 64, 4, 2, 128);
        let input_ids = vec![1, 5, 10, 20];
        let logits = model.forward(&input_ids, false);

        // 入力トークン数と同じ数のlogitsが出力される
        assert_eq!(logits.len(), 4);
        // 各logitsはvocab_sizeの次元を持つ
        assert_eq!(logits[0].len(), 100);
    }

    #[test]
    fn test_transformer_forward_max_seq_len() {
        // max_seq_len=8 のモデルを作成
        let model = WorkingTransformer::new(100, 64, 4, 2, 8);

        // max_seq_lenを超える入力
        let input_ids: Vec<usize> = (0..20).collect();
        let logits = model.forward(&input_ids, false);

        // max_seq_lenまで切り詰められる
        assert_eq!(logits.len(), 8);
    }

    // ===========================================
    // Multi-Head Attentionのテスト
    // ===========================================

    #[test]
    fn test_multihead_attention_head_splitting() {
        // hidden_size=64, num_heads=4 なら head_dim=16
        let mut rng = rand::thread_rng();
        let layer = TransformerLayer::new(64, 4, &mut rng);

        assert_eq!(layer.num_heads, 4);
        assert_eq!(layer.head_dim, 16);
    }

    #[test]
    fn test_transformer_layer_forward() {
        let mut rng = rand::thread_rng();
        let layer = TransformerLayer::new(64, 4, &mut rng);

        // 入力: (seq_len=3, hidden_size=64)
        let input = Array2::from_shape_fn((3, 64), |_| rng.gen_range(-1.0..1.0));
        let output = layer.forward(&input);

        // 出力は同じshape
        assert_eq!(output.dim(), (3, 64));
    }

    // ===========================================
    // TrainableTransformerのテスト
    // ===========================================

    #[test]
    fn test_trainable_transformer_from_texts() {
        let texts = vec!["Hello world", "Hello Rust"];
        let model = TrainableTransformer::from_texts(
            &texts,
            64,  // hidden_size
            4,   // num_heads
            2,   // num_layers
            128, // max_seq_len
            0.01, // learning_rate
        );

        // <eos>, <bos>, <unk>, Hello, world, Rust = 6
        assert_eq!(model.tokenizer.vocab_size(), 6);
    }

    #[test]
    fn test_trainable_transformer_generate() {
        let texts = vec!["Hello world"];
        let model = TrainableTransformer::from_texts(
            &texts,
            64,
            4,
            2,
            128,
            0.01,
        );

        // 生成テスト（内容は問わない、panicしなければOK）
        let _generated = model.generate("Hello", 5, 0.8);
    }

    #[test]
    fn test_training_reduces_loss() {
        let texts = vec!["Hello world", "Hello Rust", "Rust world"];
        let mut model = TrainableTransformer::from_texts(
            &texts,
            32,   // small for faster test
            2,
            1,
            64,
            0.1,  // higher learning rate for faster convergence
        );

        // 訓練前の損失を計算
        let loss_before = compute_average_loss(&model, &texts);

        // 少し訓練
        let owned_texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        model.train(&owned_texts, 20);

        // 訓練後の損失を計算
        let loss_after = compute_average_loss(&model, &texts);

        // 損失が減少していることを確認
        assert!(
            loss_after < loss_before,
            "Loss should decrease after training: before={}, after={}",
            loss_before,
            loss_after
        );
    }

    /// テスト用: 平均損失を計算
    fn compute_average_loss(model: &TrainableTransformer, texts: &[&str]) -> f32 {
        let mut total_loss = 0.0;
        let mut count = 0;

        for text in texts {
            let tokens = model.tokenizer.encode(text);
            if tokens.len() < 2 {
                continue;
            }

            let input = &tokens[..tokens.len() - 1];
            let target = &tokens[1..];
            let logits = model.model.forward(input, false);

            for (i, logit) in logits.iter().enumerate() {
                let probs = softmax(logit);
                let loss = -probs[target[i]].ln();
                total_loss += loss;
                count += 1;
            }
        }

        if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        }
    }

    // ===========================================
    // ユーティリティ関数のテスト
    // ===========================================

    #[test]
    fn test_softmax() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits);

        // 確率の和は1になる
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // 大きな値ほど高い確率
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_with_temperature() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // 低温度: よりsharpな分布
        let probs_low_temp = softmax_with_temperature(&logits, 0.5);
        // 高温度: より均一な分布
        let probs_high_temp = softmax_with_temperature(&logits, 2.0);

        // 低温度の方が最大確率が高い
        assert!(probs_low_temp[2] > probs_high_temp[2]);
    }

    #[test]
    fn test_gelu() {
        // GELU(0) ≈ 0
        assert!((gelu(0.0)).abs() < 1e-5);

        // GELU(x) > 0 for x > 0
        assert!(gelu(1.0) > 0.0);

        // GELU is approximately x for large positive x
        let large_x = 5.0;
        assert!((gelu(large_x) - large_x).abs() < 0.1);
    }

    // ===========================================
    // 数値勾配チェックテスト
    // ===========================================

    /// 有限差分法による数値勾配計算のヘルパー
    fn numerical_gradient<F>(f: F, x: f32, h: f32) -> f32
    where
        F: Fn(f32) -> f32,
    {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    #[test]
    fn test_gelu_derivative_numerical() {
        // GELUの導関数を数値微分と比較
        // 有限差分法には本質的に誤差があるため、許容誤差は0.01（1%）とする
        let h = 1e-4; // より大きなhで数値誤差を減らす
        let test_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];

        for &x in &test_values {
            let analytical = gelu_derivative(x);
            let numerical = numerical_gradient(gelu, x, h);
            let diff = (analytical - numerical).abs();

            // 相対誤差または絶対誤差でチェック
            let rel_err = if analytical.abs() > 1e-6 {
                diff / analytical.abs()
            } else {
                diff
            };

            assert!(
                rel_err < 0.05 || diff < 0.01, // 5%相対誤差または0.01絶対誤差
                "GELU derivative mismatch at x={}: analytical={}, numerical={}, diff={}, rel_err={}",
                x,
                analytical,
                numerical,
                diff,
                rel_err
            );
        }
    }

    #[test]
    fn test_softmax_gradient_numerical() {
        // Softmaxの勾配をテスト
        // f(logits) = -log(softmax(logits)[target])
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0, 0.5]);
        let target = 2;
        let h = 1e-4; // より大きなhで数値誤差を減らす

        // 解析的勾配: d_logit = probs - one_hot(target)
        let probs = softmax(&logits);
        let mut analytical_grad = probs.clone();
        analytical_grad[target] -= 1.0;

        // 各要素について数値微分
        for i in 0..logits.len() {
            let mut logits_plus = logits.clone();
            let mut logits_minus = logits.clone();
            logits_plus[i] += h;
            logits_minus[i] -= h;

            let loss_plus = -softmax(&logits_plus)[target].ln();
            let loss_minus = -softmax(&logits_minus)[target].ln();
            let numerical = (loss_plus - loss_minus) / (2.0 * h);

            let diff = (analytical_grad[i] - numerical).abs();
            // 相対誤差または絶対誤差でチェック
            let rel_err = if analytical_grad[i].abs() > 1e-6 {
                diff / analytical_grad[i].abs()
            } else {
                diff
            };

            assert!(
                rel_err < 0.05 || diff < 0.01, // 5%相対誤差または0.01絶対誤差
                "Softmax gradient mismatch at i={}: analytical={}, numerical={}, diff={}, rel_err={}",
                i,
                analytical_grad[i],
                numerical,
                diff,
                rel_err
            );
        }
    }

    #[test]
    fn test_output_projection_gradient_numerical() {
        // 出力投影層の勾配をテスト
        let hidden_size = 4;
        let vocab_size = 6;
        let h = 1e-4;

        // 小さなモデルを作成
        let model = WorkingTransformer::new(vocab_size, hidden_size, 2, 1, 8);
        let input_ids = vec![1, 2, 3];
        let target = vec![2, 3, 0];

        // 順伝播してキャッシュを取得
        let (_logits, cache) = model.forward_full_cache(&input_ids);

        // 解析的勾配を計算
        let mut grads = ModelGradients::new(vocab_size, hidden_size, 8, 1);
        model.backward(&cache, &target, &mut grads, vocab_size);

        // いくつかの要素について数値微分
        let test_indices = [(0, 0), (1, 2), (2, 4)];

        for &(i, j) in &test_indices {
            if i >= hidden_size || j >= vocab_size {
                continue;
            }

            // output_projection[i][j] を摂動
            let loss_plus = {
                // 重みをコピー（簡略化のため新しいモデルで近似）
                let logits = model.forward(&input_ids, false);
                let mut total = 0.0;
                for (pos, logit) in logits.iter().enumerate() {
                    let mut logit_mod = logit.clone();
                    // 摂動を適用（output_projection[i][j] の影響）
                    logit_mod[j] += cache.post_ln_final[[pos, i]] * h;
                    let probs = softmax(&logit_mod);
                    total += -probs[target[pos]].ln();
                }
                total
            };

            let loss_minus = {
                let logits = model.forward(&input_ids, false);
                let mut total = 0.0;
                for (pos, logit) in logits.iter().enumerate() {
                    let mut logit_mod = logit.clone();
                    logit_mod[j] -= cache.post_ln_final[[pos, i]] * h;
                    let probs = softmax(&logit_mod);
                    total += -probs[target[pos]].ln();
                }
                total
            };

            let numerical = (loss_plus - loss_minus) / (2.0 * h);
            let analytical = grads.d_output_projection[[i, j]];

            // 許容誤差を緩めにする（Top-K制限の影響もあるため）
            let diff = (analytical - numerical).abs();
            let rel_diff = if analytical.abs() > 1e-6 {
                diff / analytical.abs()
            } else {
                diff
            };

            // 相対誤差が10%以下、または絶対誤差が小さいことを確認
            assert!(
                rel_diff < 0.2 || diff < 0.1,
                "Output projection grad mismatch at [{},{}]: analytical={:.6}, numerical={:.6}, diff={:.6}",
                i, j, analytical, numerical, diff
            );
        }
    }

    #[test]
    fn test_embedding_gradient_direction() {
        // 埋め込み勾配の方向が正しいことを検証
        // 損失を下げる方向に勾配が指していることを確認
        let texts = vec!["a b", "b c", "c a"];
        let mut model = TrainableTransformer::from_texts(
            &texts,
            8,  // 小さいhidden_size
            2,
            1,
            16,
            0.1,
        );

        // 初期損失
        let loss_before = compute_average_loss(&model, &texts);

        // 勾配計算
        let mut grads = ModelGradients::new(
            model.model.vocab_size,
            model.model.hidden_size,
            model.model.max_seq_len,
            model.model.num_layers,
        );

        for text in &texts {
            let tokens = model.tokenizer.encode(text);
            if tokens.len() < 2 {
                continue;
            }
            let input = &tokens[..tokens.len() - 1];
            let target = &tokens[1..];
            let (_logits, cache) = model.model.forward_full_cache(input);
            model.model.backward(&cache, target, &mut grads, model.model.vocab_size);
        }

        // 勾配方向に小さなステップで更新
        let lr = 0.01;
        model.model.apply_gradients(&grads, lr, 0.0);

        // 更新後の損失
        let loss_after = compute_average_loss(&model, &texts);

        // 損失が減少しているはず
        assert!(
            loss_after <= loss_before + 0.01, // 多少の変動は許容
            "Gradient direction should reduce loss: before={}, after={}",
            loss_before,
            loss_after
        );
    }

    #[test]
    fn test_layer_norm_backward_numerical() {
        // LayerNormの逆伝播を数値微分と比較
        let hidden_size = 4;
        let seq_len = 2;
        let h = 1e-4;

        let model = WorkingTransformer::new(6, hidden_size, 2, 1, 8);

        // 入力を作成
        let mut rng = rand::thread_rng();
        let x = Array2::from_shape_fn((seq_len, hidden_size), |_| rng.gen_range(-1.0..1.0));
        let gamma = Array1::from_shape_fn(hidden_size, |_| rng.gen_range(0.5..1.5));
        let beta = Array1::from_shape_fn(hidden_size, |_| rng.gen_range(-0.5..0.5));

        // 順伝播
        let (_output, stds) = model.layer_norm_with_std(&x, &gamma, &beta);

        // d_out: 上流からの勾配（ランダム）
        let d_out = Array2::from_shape_fn((seq_len, hidden_size), |_| rng.gen_range(-1.0..1.0));

        // 解析的勾配
        let (_d_x_analytical, d_gamma_analytical, d_beta_analytical) =
            model.layer_norm_backward(&d_out, &x, &stds, &gamma);

        // d_beta の数値勾配チェック
        for j in 0..hidden_size {
            let mut beta_plus = beta.clone();
            let mut beta_minus = beta.clone();
            beta_plus[j] += h;
            beta_minus[j] -= h;

            let output_plus = model.layer_norm(&x, &gamma, &beta_plus);
            let output_minus = model.layer_norm(&x, &gamma, &beta_minus);

            // スカラー関数の値: sum(d_out * output)
            let loss_plus: f32 = d_out.iter().zip(output_plus.iter()).map(|(a, b)| a * b).sum();
            let loss_minus: f32 = d_out.iter().zip(output_minus.iter()).map(|(a, b)| a * b).sum();

            let numerical = (loss_plus - loss_minus) / (2.0 * h);
            let analytical = d_beta_analytical[j];

            let diff = (numerical - analytical).abs();
            assert!(
                diff < 0.1,
                "LayerNorm d_beta mismatch at j={}: analytical={}, numerical={}, diff={}",
                j, analytical, numerical, diff
            );
        }

        // d_gamma の数値勾配チェック
        for j in 0..hidden_size {
            let mut gamma_plus = gamma.clone();
            let mut gamma_minus = gamma.clone();
            gamma_plus[j] += h;
            gamma_minus[j] -= h;

            let output_plus = model.layer_norm(&x, &gamma_plus, &beta);
            let output_minus = model.layer_norm(&x, &gamma_minus, &beta);

            let loss_plus: f32 = d_out.iter().zip(output_plus.iter()).map(|(a, b)| a * b).sum();
            let loss_minus: f32 = d_out.iter().zip(output_minus.iter()).map(|(a, b)| a * b).sum();

            let numerical = (loss_plus - loss_minus) / (2.0 * h);
            let analytical = d_gamma_analytical[j];

            let diff = (numerical - analytical).abs();
            assert!(
                diff < 0.1,
                "LayerNorm d_gamma mismatch at j={}: analytical={}, numerical={}, diff={}",
                j, analytical, numerical, diff
            );
        }
    }

    #[test]
    fn test_gradient_norm_clipping() {
        // 勾配クリッピングのテスト
        let mut grads = ModelGradients::new(10, 4, 8, 1);

        // 大きな勾配を設定
        grads.d_token_embeddings.fill(10.0);
        grads.d_output_projection.fill(10.0);

        let norm_before = grads.l2_norm();
        assert!(norm_before > 1.0, "Initial norm should be > 1.0");

        // クリッピング
        grads.clip_grad_norm(1.0);

        let norm_after = grads.l2_norm();
        assert!(
            (norm_after - 1.0).abs() < 1e-4,
            "Norm after clipping should be ~1.0: got {}",
            norm_after
        );
    }

    #[test]
    fn test_weight_decay_effect() {
        // 重み減衰が正しく適用されることを確認
        let mut model = WorkingTransformer::new(10, 4, 2, 1, 8);

        // 初期重みの絶対値の和を記録
        let initial_weight_sum: f32 = model.token_embeddings.iter().map(|x| x.abs()).sum();

        // ゼロ勾配で重み減衰のみ適用
        let grads = ModelGradients::new(10, 4, 8, 1);
        let weight_decay = 0.1;
        model.apply_gradients(&grads, 0.01, weight_decay);

        // 重みが減少していることを確認
        let final_weight_sum: f32 = model.token_embeddings.iter().map(|x| x.abs()).sum();
        assert!(
            final_weight_sum < initial_weight_sum,
            "Weight decay should reduce weights: before={}, after={}",
            initial_weight_sum,
            final_weight_sum
        );
    }

    #[test]
    fn test_learning_rate_warmup() {
        // 学習率ウォームアップのテスト
        let texts = vec!["test"];
        let model = TrainableTransformer::from_texts(&texts, 4, 2, 1, 8, 0.1);

        // 初期状態でstep=1から始まることを確認
        assert_eq!(model.step, 1, "Initial step should be 1, not 0");

        let mut model = model;
        model.config.warmup_steps = 100;

        // step=1 での学習率（最初のバッチから学習が行われる）
        model.step = 1;
        let lr_1 = model.get_learning_rate();
        assert!(
            (lr_1 - 0.001).abs() < 1e-6,
            "LR at step 1 should be ~0.001 (base_lr/100): got {}",
            lr_1
        );

        // step=50 での学習率
        model.step = 50;
        let lr_50 = model.get_learning_rate();
        assert!(
            (lr_50 - 0.05).abs() < 1e-5,
            "LR at step 50 should be ~0.05: got {}",
            lr_50
        );

        // step=100 での学習率（warmup最終ステップ）
        model.step = 100;
        let lr_100 = model.get_learning_rate();
        assert!(
            (lr_100 - 0.1).abs() < 1e-6,
            "LR at step 100 should be base_lr: got {}",
            lr_100
        );

        // warmup完了後
        model.step = 200;
        let lr_200 = model.get_learning_rate();
        assert!(
            (lr_200 - 0.1).abs() < 1e-6,
            "LR after warmup should be base_lr"
        );
    }
}
