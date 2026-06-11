from math import sqrt

import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, PatchEmbedding
from transformers import (
    LlamaConfig, LlamaModel, AutoTokenizer,
    GPT2Config, GPT2Model, GPT2Tokenizer,
    BertConfig, BertModel, BertTokenizer,
)
import transformers
from layers.StandardNorm import Normalize

# Suppress verbose HuggingFace loading messages
transformers.logging.set_verbosity_error()


# ─────────────────────────────────────────────────────────────────────────────
# FlattenHead
# Component ⑥ (Output Projection) — flattens LLM patch outputs and projects
# them to the desired forecast horizon length.
# Input:  (batch, n_vars, d_ff, patch_nums)
# Output: (batch, n_vars, pred_len + label_len)
# ─────────────────────────────────────────────────────────────────────────────
class FlattenHead(nn.Module):

    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars  = n_vars
        self.flatten = nn.Flatten(start_dim=-2)          # merge last two dims
        self.linear  = nn.Linear(nf, target_window)      # project to forecast length
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# CrossAttentionLayer
# Component ② — bridges time-series patches and LLM word embeddings.
# Queries come from the time-series patches (d_model).
# Keys and Values come from the LLM vocabulary embeddings (d_llm).
# This converts time-series data into a format the LLM can process.
# ─────────────────────────────────────────────────────────────────────────────
class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(CrossAttentionLayer, self).__init__()

        # if d_keys not given, split d_model evenly across heads
        d_keys = d_keys or (d_model // n_heads)

        # Q comes from time-series; K and V come from LLM vocabulary
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection   = nn.Linear(d_llm,   d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm,   d_keys * n_heads)
        self.out_projection   = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads          = n_heads
        self.dropout          = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        # target_embedding: (batch, num_patches, d_model)  — from time series
        # source_embedding: (num_tokens, d_llm)            — LLM word keys
        # value_embedding:  (num_tokens, d_llm)            — LLM word values
        B, T, N = target_embedding.shape
        S, _    = source_embedding.shape
        H       = self.n_heads

        # project and split into multiple heads
        target_embedding = self.query_projection(target_embedding).view(B, T, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding  = self.value_projection(value_embedding).view(S, H, -1)

        # compute scaled dot-product attention
        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        # merge heads and project to LLM dimension
        out = out.reshape(B, T, -1)
        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        # Equations (3) and (4) from the paper
        B, L, H, E = target_embedding.shape
        scale = 1. / sqrt(E)

        # attention scores: how much each patch attends to each token
        # einsum "blhe,she->bhls": (batch, patches, heads, dim) x (tokens, heads, dim)
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        # softmax over tokens dimension to get attention weights
        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        # weighted sum of value embeddings
        # einsum "bhls,she->blhe": (batch, heads, patches, tokens) x (tokens, heads, dim)
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)
        return reprogramming_embedding


# ─────────────────────────────────────────────────────────────────────────────
# LLMBlock
# Components ①②③ — the full LLM pipeline for target features.
# 1. Word Projection   ①: compress LLM vocabulary from ~50k to num_tokens
# 2. Patch Embedding      : split time series into overlapping patches
# 3. Cross-Attention   ②: align patches with LLM token space
# 4. Frozen LLM        ③: process reprogrammed embeddings (weights frozen or partially updated)
# 5. Output Projection    : reshape and project back to forecast horizon
# ─────────────────────────────────────────────────────────────────────────────
class LLMBlock(nn.Module):

    def __init__(self, configs):
        super(LLMBlock, self).__init__()

        # store config values needed in forward pass
        self.device    = configs.device
        self.task_name = configs.task_name
        self.pred_len  = configs.pred_len
        self.seq_len   = configs.seq_len
        self.d_ff      = configs.d_ff
        self.top_k     = configs.top_k
        self.d_llm     = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride    = configs.stride

        self.use_prompt   = configs.use_prompt    # always False for MultiAttLLM
        self.use_forecast = configs.use_forecast

        # ── Load pre-trained LLM backbone ────────────────────────────────────
        # Each branch loads the config, sets the number of layers to use,
        # then loads the model weights from HuggingFace.
        # try/except pattern: attempts load, falls back with same call (for
        # compatibility in case of network issues).

        if configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers  = configs.llm_layers
            self.llama_config.output_attentions  = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                )
            except EnvironmentError:
                print("Model not found locally. Downloading...")
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    'huggyllama/llama-7b', trust_remote_code=True, local_files_only=False)
            except EnvironmentError:
                print("Tokenizer not found locally. Downloading...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    'huggyllama/llama-7b', trust_remote_code=True, local_files_only=False)

        elif configs.llm_model == 'GPT2':
            # GPT2 is the paper's chosen backbone (best speed/accuracy tradeoff)
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers   = configs.llm_layers  # use only 6 of 12
            self.gpt2_config.output_attentions   = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )
            except EnvironmentError:
                print("Model not found locally. Downloading...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2', trust_remote_code=True, local_files_only=False)
            except EnvironmentError:
                print("Tokenizer not found locally. Downloading...")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2', trust_remote_code=True, local_files_only=False)

        elif configs.llm_model == 'QWEN':
            self.bert_config = BertConfig.from_pretrained('Qwen/Qwen2-7B-Instruct')
            self.bert_config.num_hidden_layers   = configs.llm_layers
            self.bert_config.output_attentions   = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'Qwen/Qwen2-7B-Instruct',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
            except EnvironmentError:
                print("Model not found locally. Downloading...")
                self.llm_model = BertModel.from_pretrained(
                    'Qwen/Qwen2-7B-Instruct',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'Qwen/Qwen2-7B-Instruct', trust_remote_code=True, local_files_only=False)
            except EnvironmentError:
                print("Tokenizer not found locally. Downloading...")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'Qwen/Qwen2-7B-Instruct', trust_remote_code=True, local_files_only=False)

        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            self.bert_config.num_hidden_layers   = configs.llm_layers
            self.bert_config.output_attentions   = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
            except EnvironmentError:
                print("Model not found locally. Downloading...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased', trust_remote_code=True, local_files_only=False)
            except EnvironmentError:
                print("Tokenizer not found locally. Downloading...")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased', trust_remote_code=True, local_files_only=False)

        else:
            raise Exception('LLM model is not defined')

        # ensure the tokenizer has a padding token
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = '[PAD]'

        # ── Freeze all LLM weights — they are never updated during training ──
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # ── Partial LLM Fine-Tuning (optional) ────────────────────────────────
        # Unfreezes the last N transformer layers while keeping all earlier
        # layers frozen. Setting llm_finetune_layers = 0 preserves the paper's
        # original fully-frozen LLM behaviour.
        n_unfreeze = min(configs.llm_finetune_layers, configs.llm_layers)

        if n_unfreeze > 0:

            if configs.llm_model == 'GPT2':
                transformer_layers = self.llm_model.h

            elif configs.llm_model in ['BERT', 'QWEN']:
                transformer_layers = self.llm_model.encoder.layer

            elif configs.llm_model == 'LLAMA':
                transformer_layers = self.llm_model.layers

            else:
                transformer_layers = []

            # unfreeze the final N transformer blocks
            for layer in transformer_layers[-n_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True

            print(
                f"LLM partial fine-tuning enabled: "
                f"{n_unfreeze}/{len(transformer_layers)} layers unfrozen "
                f"(layers {len(transformer_layers)-n_unfreeze} "
                f"to {len(transformer_layers)-1})"
            )
        else:
            print(
                f"LLM frozen: 0/{configs.llm_layers} layers unfrozen"
            )

        # ── Trainable Parameter Summary ───────────────────────────────────────
        # Reports how many parameters remain trainable after freezing and
        # optional partial LLM fine-tuning.

        total_params = sum(
            p.numel() for p in self.llm_model.parameters()
        )

        trainable_params = sum(
            p.numel() for p in self.llm_model.parameters()
            if p.requires_grad
        )

        print(
            f"LLM trainable parameters: "
            f"{trainable_params:,}/{total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

        # ── Patch embedding: splits time series into overlapping patches ──────
        self.dropout        = nn.Dropout(configs.dropout)
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        # number of patches: formula from paper
        # e.g. (72 - 16) / 8 + 2 = 9 patches
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)

        # ── Word Projection ①: compress vocabulary from ~50k to num_tokens ───
        # configs.num_tokens is set in setup_MultiAttLLM.py (paper default: 2000)
        self.word_embeddings  = self.llm_model.get_input_embeddings().weight
        self.vocab_size       = self.word_embeddings.shape[0]
        self.num_tokens       = configs.num_tokens
        self.word_projection  = nn.Linear(self.vocab_size, self.num_tokens)

        # ── Output Projection ⑥: flatten patches and project to forecast len ─
        self.head_nf          = self.d_ff * self.patch_nums
        self.output_projection = FlattenHead(
            configs.enc_in,
            self.head_nf,
            self.pred_len + configs.label_len,
            head_dropout=configs.dropout
        )

        # ── Cross-Attention ②: aligns patches with LLM token space ───────────
        self.crossattention_layer = CrossAttentionLayer(
            configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        # instance normalisation — removes mean/std before LLM, restores after
        self.normalize_layers = Normalize(configs.enc_in, affine=False)

        # move all sub-modules to the correct device (GPU or CPU)
        self.llm_model.to(device=self.device)
        self.word_projection.to(device=self.device)
        self.crossattention_layer.to(device=self.device)
        self.patch_embedding.to(device=self.device)
        self.output_projection.to(device=self.device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # entry point — delegates to forecast()
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, :, :]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Step 1: normalise input (zero mean, unit std per instance)
        x_enc = self.normalize_layers(x_enc, 'norm')

        # Step 2: compress LLM vocabulary — Equation (1) in paper
        # permute needed because nn.Linear acts on last dim
        word_embeddings = self.word_projection(
            self.word_embeddings.permute(1, 0)   # (d_llm, vocab) → (vocab, d_llm)
        ).permute(1, 0)                           # back to (num_tokens, d_llm)

        # Step 3: convert time series to patches
        # permute to (batch, n_vars, seq_len) — required by PatchEmbedding
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        # bfloat16 reduces memory usage while keeping sufficient precision
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        # enc_out: (batch * n_vars, patch_nums, d_model)

        # Step 4: cross-attention reprogramming — Equations (2)(3)(4) in paper
        # patches attend to compressed LLM word tokens
        enc_out = self.crossattention_layer(enc_out, word_embeddings, word_embeddings)
        # enc_out: (batch * n_vars, patch_nums, d_llm)

        # Step 5: pass through frozen LLM — Equation (5) in paper
        # inputs_embeds bypasses GPT2's own embedding layer
        dec_out = self.llm_model(inputs_embeds=enc_out).last_hidden_state
        # keep only first d_ff dimensions of LLM output (discard the rest)
        dec_out = dec_out[:, :, :self.d_ff]

        # Step 6: reshape and project to forecast horizon
        # reshape back to (batch, n_vars, d_ff, patch_nums)
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        # FlattenHead projects (batch, n_vars, d_ff, patch_nums) → (batch, n_vars, pred+label)
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        # permute back to (batch, pred+label, n_vars)
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        # Step 7: reverse the normalisation applied in Step 1
        dec_out = self.normalize_layers(dec_out, 'denorm')
        return dec_out

    def calcute_lags(self, x_enc):
        # computes top-k autocorrelation lags via FFT (Wiener-Khinchin theorem)
        # used for prompt statistics — not part of the main forward pass
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res   = q_fft * torch.conj(k_fft)
        corr  = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


# ─────────────────────────────────────────────────────────────────────────────
# Model (MultiAttLLM)
# The top-level model — ties all components together.
# Two encoder paths run in parallel:
#   Path A (LLM encoder)       : target features → LLMBlock → enc_out_target
#   Path B (covariate encoder) : non-target features → feature_extractor → enc_out_other
# Then a decoder fuses both via cross-attention and projects to final predictions.
# ─────────────────────────────────────────────────────────────────────────────
class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name        = configs.task_name
        self.pred_len         = configs.pred_len
        self.c_out            = configs.c_out          # number of target variables (3)
        self.output_attention = configs.output_attention
        self.use_forecast     = configs.use_forecast

        # number of non-target (covariate) features
        # e.g. 22 total - 3 targets = 19 covariates
        n_covariates          = configs.enc_in - self.c_out
        self.extractor_type   = configs.feature_extractor_type

        # ── Component ④: Covariate Feature Extractor ─────────────────────────
        # Selectable via configs.feature_extractor_type in setup_MultiAttLLM.py
        # Paper default: 'linear' (simple linear layer, Section 2.2)

        if self.extractor_type == 'linear':
            # simplest option — one linear layer maps covariates to d_model
            # input:  (batch, seq_len, n_covariates)
            # output: (batch, seq_len, d_model)
            self.feature_extractor = nn.Linear(n_covariates, configs.d_model)

        elif self.extractor_type == 'lstm':
            # RNN-based extractor — captures temporal patterns in covariates
            # input:  (batch, seq_len, n_covariates)
            # output: (batch, seq_len, d_model)
            self.feature_extractor = nn.LSTM(
                input_size=n_covariates,
                hidden_size=configs.d_model,
                num_layers=2,
                batch_first=True,       # expect (batch, seq, features) input
                dropout=configs.dropout,
                bidirectional=False
            )

        elif self.extractor_type == 'transformer':
            # transformer-based extractor — uses self-attention on covariates
            # needs DataEmbedding first to add positional/time encoding
            # input:  (batch, seq_len, n_covariates)
            # output: (batch, seq_len, d_model)
            self.enc_embedding = DataEmbedding(
                n_covariates, configs.d_model,
                configs.embed, configs.freq, configs.dropout)
            self.feature_extractor = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(
                                False, configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=configs.output_attention),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        4 * configs.d_model,    # feedforward hidden dim = 4 * d_model
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.e_layers)   # e_layers = 1
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )

        # ── Component ③: LLM Encoder (target features only) ──────────────────
        # LLMBlock handles Components ①②③ internally
        self.LLM_encoder = LLMBlock(configs)

        # ── Component ⑤: Fusion Decoder (Self-Attention) ─────────────────────
        # Takes LLM output as input, cross-attends to covariate encoding,
        # producing a fused representation for final prediction.
        # d_layers = 4 decoder layers (paper Table 3)
        self.dec_embedding = DataEmbedding(
            configs.c_out, configs.d_model,
            configs.embed, configs.freq, configs.dropout)

        self.selfattention_layer = Decoder(
            [
                DecoderLayer(
                    # self-attention on decoder sequence (causal — mask=True)
                    AttentionLayer(
                        FullAttention(True, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    # cross-attention between decoder and covariate encoder output
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    4 * configs.d_model,    # feedforward hidden dim
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)    # d_layers = 4
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # ── Component ⑥: Output Projection ───────────────────────────────────
        # maps d_model → c_out (number of target variables)
        self.out_projection  = nn.Linear(configs.d_model, configs.c_out)

        # auxiliary linear layer (not used in main forward path)
        self.linear_predict  = nn.Linear(
            configs.seq_len, configs.pred_len + configs.label_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # ── Split input features into target and covariate groups ─────────────
        # data_loader ensures target columns are always placed last
        x_enc_other  = x_enc[:, :, :-self.c_out]   # covariates: (batch, 72, 19)
        x_enc_target = x_enc[:, :, -self.c_out:]   # targets:    (batch, 72, 3)

        # ── Path A: target features through LLM ───────────────────────────────
        enc_out_target = self.LLM_encoder(x_enc_target, x_mark_enc, x_dec, x_mark_dec)
        # output: (batch, pred_len + label_len, c_out) = (batch, 240, 3)

        # ── Path B: covariate features through selected extractor ─────────────
        if self.extractor_type == 'linear':
            enc_out_other = self.feature_extractor(x_enc_other)
            # (batch, 72, 19) → (batch, 72, d_model)

        elif self.extractor_type == 'lstm':
            enc_out_other, _ = self.feature_extractor(x_enc_other)
            # _ discards the (hidden_state, cell_state) tuple — we only need output
            # (batch, 72, 19) → (batch, 72, d_model)

        elif self.extractor_type == 'transformer':
            enc_out_other, _ = self.feature_extractor(
                self.enc_embedding(x_enc_other, x_mark_enc))
            # embed first (adds time features), then encode with self-attention
            # _ discards attention weights — we only need the encoded output
            # (batch, 72, 19) → embed → (batch, 72, d_model) → encode → (batch, 72, d_model)

        # ── Fusion: decoder cross-attends LLM output to covariate encoding ────
        # embed the LLM output to add time encoding before passing to decoder
        dec_in  = self.dec_embedding(enc_out_target, x_mark_dec)
        # (batch, 240, 3) → (batch, 240, d_model)

        dec_out = self.selfattention_layer(
            dec_in, enc_out_other, x_mask=None, cross_mask=None)
        # decoder self-attends on dec_in, cross-attends to enc_out_other
        # output: (batch, 240, d_model)

        # ── Final projection to target variable dimension ─────────────────────
        dec_out = self.out_projection(dec_out)
        # (batch, 240, d_model) → (batch, 240, c_out=3)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # keep only the last pred_len steps — discard the label_len warm-up prefix
        # e.g. (batch, 240, 3) → (batch, 168, 3)
        return dec_out[:, -self.pred_len:, :]