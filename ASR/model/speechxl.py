import torch
import torch.nn as nn
import torch.nn.functional as F


def parallel_shift(x):
    x_size = x.size()
    x = F.pad(x, (0, 0, 0, 0, 1, 0))
    x = torch.reshape(x, (x_size[1] + 1, x_size[0], x_size[2], x_size[3]))
    x = x[1:]
    x = torch.reshape(x, x_size)

    return x

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]

class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inp):
        ##### positionwise feed-forward
        core_out = self.CoreNet(inp)

        ##### residual connection + layer normalization
        output = self.layer_norm(inp + core_out)

        return output

class RelMultiHeadAttn(nn.Module):
    def __init__(self, mode, n_head, d_model, d_head, dropout, dropatt, chunk_size=100):
        super(RelMultiHeadAttn, self).__init__()

        # mode=0: enc_self_att
        # mode=1: enc_dec_att
        # mode=2: dec_self_att
        # mode=3: fast_enc_self_att
        self.mode = mode

        self.n_head = n_head
        self.d_head = d_head
        self.dropout = dropout
        self.chunk_size = chunk_size

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        if self.mode == 0 or self.mode == 3:
            self.r_net_fw = nn.Linear(d_model, n_head * d_head, bias=False)
            self.r_net_bw = nn.Linear(d_model, n_head * d_head, bias=False)
        elif self.mode == 2:
            self.r_net = nn.Linear(d_model, n_head * d_head, bias=False)
        elif self.mode == 1:
            # no positional encodings for enc_dec_att
            pass
        else:
            assert False

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

    def forward(self, *args, **kwargs):
        if self.mode == 0:
            return self.enc_self_att(*args, **kwargs)
        elif self.mode == 1:
            return self.enc_dec_att(*args, **kwargs)
        elif self.mode == 2:
            return self.dec_self_att(*args, **kwargs)
        elif self.mode == 3:
            if args[0].size(0) <= self.chunk_size:
                return self.enc_self_att(*args, **kwargs)
            return self.fast_enc_self_att_v2(*args, **kwargs)
        else:
            assert False

    def attention(self, layer_input, head_v, ac_head_q, ac_head_k,
            bd_head_q=None, bd_head_k=None, shift_func=None, attn_mask=None):
        AC = torch.einsum('ibnd,jbnd->ijbn', [ac_head_q, ac_head_k])

        if bd_head_q is not None:
            BD = torch.einsum('ibnd,jnd->ijbn', [bd_head_q, bd_head_k])
            BD = shift_func(BD)
            attn_score = AC + BD
        else:
            attn_score = AC

        attn_score.mul_(self.scale)

        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -1e30)
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -1e30)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        ret_attn_prob = attn_prob
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        ##### residual connection + layer normalization
        output = self.layer_norm(layer_input + attn_out)

        return output,ret_attn_prob

    def enc_self_att(self, w, r, r_w_bias, r_r_bias, attn_mask=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        w_heads = self.qkv_net(w)
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        r_head_k_fw = self.r_net_fw(r)
        r_head_k_bw = self.r_net_bw(r)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(qlen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(qlen, bsz, self.n_head, self.d_head)

        r_head_k_fw = r_head_k_fw.view(rlen, self.n_head, self.d_head)
        r_head_k_bw = r_head_k_bw.view(rlen, self.n_head, self.d_head)
        # remove duplicated position 0
        r_head_k = torch.cat([r_head_k_fw, torch.flip(r_head_k_bw, (0, ))[1:]], 0)

        ac_head_q = w_head_q + r_w_bias
        ac_head_k = w_head_k
        bd_head_q = w_head_q + r_r_bias
        bd_head_k = r_head_k

        def shift(BD):
            qlen = BD.size(0)
            return parallel_shift(BD)[:, :qlen]
            # return ParallelShift(qlen)(BD)[:, :qlen]

        return self.attention(w, w_head_v, ac_head_q, ac_head_k,
            bd_head_q, bd_head_k, shift, attn_mask)

    def fast_enc_self_att_v2(self, w, r, r_w_bias, r_r_bias, attn_mask=None):
        bsz = w.size(1)

        w_heads = self.qkv_net(w)
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        r_head_k_fw = self.r_net_fw(r)
        r_head_k_bw = self.r_net_bw(r)

        i = 0
        all_values = []
        chunk_size = self.chunk_size
        while i < w.size(0):
            end = min(w.size(0), i + chunk_size * 2)
            start = max(0, i - chunk_size)

            my_start = i
            my_end = min(end, i + chunk_size)

            cur_w_head_q = w_head_q[my_start: my_end]
            cur_w_head_k, cur_w_head_v = w_head_k[start: end], w_head_q[start: end]
            fw_attn_size = my_end - start
            cur_r_head_k_fw = r_head_k_fw[- fw_attn_size:]
            bw_attn_size = end - my_start
            cur_r_head_k_bw = r_head_k_bw[- bw_attn_size:]
            cur_r_head_k = torch.cat([cur_r_head_k_fw,
                torch.flip(cur_r_head_k_bw, (0, ))[1:]], 0)

            qlen = my_end - my_start
            klen = end - start
            rlen = cur_r_head_k.size(0)
            cur_w_head_q = cur_w_head_q.view(qlen, bsz, self.n_head, self.d_head)
            cur_w_head_k = cur_w_head_k.view(klen, bsz, self.n_head, self.d_head)
            cur_w_head_v = cur_w_head_v.view(klen, bsz, self.n_head, self.d_head)
            cur_r_head_k = cur_r_head_k.view(rlen, self.n_head, self.d_head)

            ac_head_q = cur_w_head_q + r_w_bias
            ac_head_k = cur_w_head_k
            bd_head_q = cur_w_head_q + r_r_bias
            bd_head_k = cur_r_head_k

            def shift(BD):
                qlen = BD.size(0)
                return parallel_shift(BD)[:, :klen]
                # return ParallelShift(qlen)(BD)[:, :klen]

            all_values.append(self.attention(w[my_start: my_end], cur_w_head_v,
                ac_head_q, ac_head_k, bd_head_q, bd_head_k, shift,
                attn_mask[start: end])[0])

            i += chunk_size
        return torch.cat(all_values, 0)

    def fast_enc_self_att(self, w, r, r_w_bias, r_r_bias, attn_mask=None):
        i = 0
        all_values = []
        chunk_size = self.chunk_size
        while i < w.size(0):
            end = min(w.size(0), i + chunk_size * 2)
            start = max(0, i - chunk_size)

            my_start = i - start
            my_end = min(end - start, my_start + chunk_size)

            cur_val = self.enc_self_att(w[start: end], r[-(end-start):],
                r_w_bias, r_r_bias, attn_mask[start: end])

            all_values.append(cur_val[my_start: my_end])
            i += chunk_size
        return torch.cat(all_values, 0)

    def enc_dec_att(self, w, r, r_w_bias, attn_mask=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        w_heads = self.qkv_net(w)
        r_heads = self.qkv_net(r)

        w_head_q, _, _ = torch.chunk(w_heads, 3, dim=-1)
        _, r_head_k, r_head_v = torch.chunk(r_heads, 3, dim=-1)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        r_head_k = r_head_k.view(rlen, bsz, self.n_head, self.d_head)
        r_head_v = r_head_v.view(rlen, bsz, self.n_head, self.d_head)

        ac_head_q = w_head_q + r_w_bias
        ac_head_k = r_head_k

        return self.attention(w, r_head_v, ac_head_q, ac_head_k, attn_mask=attn_mask)

    def dec_self_att(self, w, r, r_w_bias, r_r_bias, attn_mask=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        w_heads = self.qkv_net(w)
        r_head_k = self.r_net(r)

        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(qlen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(qlen, bsz, self.n_head, self.d_head)

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)

        ac_head_q = w_head_q + r_w_bias
        ac_head_k = w_head_k
        bd_head_q = w_head_q + r_r_bias
        bd_head_k = r_head_k

        def shift(BD):
            qlen = BD.size(0)
            return parallel_shift(BD)
            # return ParallelShift(qlen)(BD)

        return self.attention(w, w_head_v, ac_head_q, ac_head_k,
            bd_head_q, bd_head_k, shift, attn_mask)

    def decode_single_step(self, w, r, r_w_bias, r_r_bias, cache_k_states,
            cache_v_states):
        # w: [beam, bsz, d]
        # r: [len, 1, d]
        # cache_k_states: [beam, hlen, bsz, d] or None
        # cache_v_states: [beam, hlen, bsz, d] or None

        if cache_k_states is None:
            hlen = 0
        else:
            hlen = cache_k_states.size(1)
        beam_size, bsz = w.size(0), w.size(1)

        w = w.view(1, beam_size * bsz, -1)
        if hlen > 0:
            cache_k_states = cache_k_states.permute(1, 0, 2, 3).contiguous().view(
                hlen, beam_size * bsz, -1)
            cache_v_states = cache_v_states.permute(1, 0, 2, 3).contiguous().view(
                hlen, beam_size * bsz, -1)

        w_heads = self.qkv_net(w)
        r_head_k = self.r_net(r[-(hlen + 1):])

        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        w_head_q = w_head_q.view(1, beam_size * bsz, self.n_head, self.d_head)
        if hlen > 0:
            cat_head_k = torch.cat([cache_k_states, w_head_k], 0)
            cat_head_v = torch.cat([cache_v_states, w_head_v], 0)
        else:
            cat_head_k, cat_head_v = w_head_k, w_head_v
        cat_head_k = cat_head_k.view(hlen + 1, beam_size * bsz, self.n_head, self.d_head)
        cat_head_v = cat_head_v.view(hlen + 1, beam_size * bsz, self.n_head, self.d_head)

        r_head_k = r_head_k.view(hlen + 1, self.n_head, self.d_head)

        ac_head_q = w_head_q + r_w_bias
        ac_head_k = cat_head_k
        bd_head_q = w_head_q + r_r_bias
        bd_head_k = r_head_k

        def shift(BD):
            return BD

        attn_result,_ = self.attention(w, cat_head_v, ac_head_q, ac_head_k,
            bd_head_q, bd_head_k, shift)
        attn_result = attn_result.view(beam_size, bsz, -1)

        cur_k_states = w_head_k.view(beam_size, bsz, -1)
        cur_v_states = w_head_v.view(beam_size, bsz, -1)

        return attn_result, cur_k_states, cur_v_states

class TransformerEncoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_head, d_inner, dropout,
            dropatt, chunk_size):
        super(TransformerEncoder, self).__init__()

        self.attn_layer = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        self.n_layer = n_layer

        mode = 0 if chunk_size <= 0 else 3
        for i in range(n_layer):
            self.attn_layer.append(RelMultiHeadAttn(
                mode, n_head, d_model, d_head, dropout, dropatt, chunk_size))
            self.ff_layers.append(PositionwiseFF(d_model, d_inner, dropout))

        self.r_w_bias = nn.Parameter(torch.Tensor(n_head, d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(n_head, d_head))

    def forward(self, x, r, attn_mask):
        output = x
        total_outputs = []

        for i in range(self.n_layer):
            output,_ = self.attn_layer[i](output, r, self.r_w_bias, self.r_r_bias, attn_mask)
            output = self.ff_layers[i](output)
            total_outputs.append(output)
        return output,total_outputs

class TransformerDecoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_head, d_inner, dropout, dropatt):
        super(TransformerDecoder, self).__init__()

        self.self_attn_layer = nn.ModuleList()
        self.enc_dec_attn_layer = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        self.n_layer = n_layer

        for i in range(n_layer):
            self.self_attn_layer.append(RelMultiHeadAttn(2, n_head, d_model,
                d_head, dropout, dropatt))
            self.enc_dec_attn_layer.append(RelMultiHeadAttn(1, n_head, d_model,
                d_head, dropout, dropatt))
            self.ff_layers.append(PositionwiseFF(d_model, d_inner,dropout))

        self.r_w_bias_self = nn.Parameter(torch.Tensor(n_head, d_head))
        self.r_r_bias_self = nn.Parameter(torch.Tensor(n_head, d_head))
        self.r_w_bias_cross = nn.Parameter(torch.Tensor(n_head, d_head))

    def forward(self, x, enc, r, self_mask, cross_mask):
        output = x
        total_outputs = []
        attention_weights = []
        for i in range(self.n_layer):
            output,_ = self.self_attn_layer[i](output, r, self.r_w_bias_self,
                self.r_r_bias_self, self_mask)
            output,attn_weights = self.enc_dec_attn_layer[i](output, enc,
                self.r_w_bias_cross, cross_mask)
            attention_weights.append(attn_weights)
            output = self.ff_layers[i](output)
            total_outputs.append(output)

        return output,total_outputs,attention_weights

    def decode_single_step(self, emb, enc, cache_k_states, cache_v_states,
            r, cross_mask):
        # emb: [beam, bsz, d]
        # cache_k_states: [beam, layer, hlen, bsz, d] or None
        # cache_v_states: [beam, layer, hlen, bsz, d] or None
        # r: [len, 1, d]

        new_k_states, new_v_states = [], []

        output = emb
        for i in range(self.n_layer):
            if cache_k_states is None:
                k_input, v_input = None, None
            else:
                k_input, v_input = cache_k_states[:, i], cache_v_states[:, i]
            output, cur_k_states, cur_v_states = self.self_attn_layer[i].decode_single_step(
                output, r, self.r_w_bias_self, self.r_r_bias_self,
                k_input, v_input)
            output,_ = self.enc_dec_attn_layer[i](output, enc,
                self.r_w_bias_cross, cross_mask)
            output = self.ff_layers[i](output)
            new_k_states.append(cur_k_states)
            new_v_states.append(cur_v_states)
        new_k_states = torch.stack(new_k_states, 1)
        new_v_states = torch.stack(new_v_states, 1)

        # output: [beam, bsz, d]
        # states: [beam, layer, bsz, d]
        return output, new_k_states, new_v_states

class Transformer(nn.Module):
    def __init__(self, n_token, enc_n_layer, dec_n_layer, n_head, d_model, d_head, d_inner,
            dropout, dropatt, tie_weight=True, clamp_len=-1, chunk_size=0,
            d_input=161, label_smooth=0.0, hw_alpha=1.0):
        super(Transformer, self).__init__()

        self.pos_emb = PositionalEmbedding(d_model)

        self.word_emb = nn.Embedding(n_token, d_model)
        self.encoder = TransformerEncoder(enc_n_layer, n_head, d_model, d_head,
            d_inner, dropout, dropatt, chunk_size)
        self.decoder = TransformerDecoder(dec_n_layer, n_head, d_model, d_head,
            d_inner, dropout, dropatt)
        self.out_layer = nn.Linear(d_model, n_token)
        self.drop = nn.Dropout(dropout)
        self.input_layer = nn.Linear(d_input, d_model)

        self.emb_scale = d_model ** 0.5
        if tie_weight:
            self.out_layer.weight = self.word_emb.weight
        self.clamp_len = clamp_len

        self.enc_n_layer = enc_n_layer
        self.dec_n_layer = dec_n_layer
        self.d_model = d_model
        self.d_head = d_head
        self.n_head = n_head
        self.n_token = n_token
        self.label_smooth = label_smooth
        self.label_smooth_t = 0.0
        self.hw_alpha = hw_alpha

    def train(self, flag=True):
        if flag:
            self.label_smooth_t = self.label_smooth
        else:
            self.label_smooth_t = 0.0
        super(Transformer, self).train(flag)

    def eval(self):
        self.label_smooth_t = 0.0
        super(Transformer, self).eval()

    def forward(self, src, tgt_input, enc_self_mask):
        src_len, tgt_len, bsz = src.size(0), tgt_input.size(0), src.size(1)

        src_pos_seq = torch.arange(src_len - 1, -1, -1.0, device=src.device)
        if self.clamp_len > 0:
            src_pos_seq.clamp_(max=self.clamp_len)
        src_pos_emb = self.pos_emb(src_pos_seq)

        src_t = self.input_layer(src)

        output,total_encode_outputs = self.encoder(self.drop(src_t), self.drop(src_pos_emb), enc_self_mask)
        tgt_emb = self.word_emb(tgt_input)
        tgt_emb.mul_(self.emb_scale)
        
        tgt_pos_seq = torch.arange(tgt_len - 1, -1, -1.0, device=src.device)
        if self.clamp_len > 0:
            tgt_pos_seq.clamp_(max=self.clamp_len)
        tgt_pos_emb = self.pos_emb(tgt_pos_seq)
        
        dec_self_mask = torch.triu(
            src.new_ones(tgt_len, tgt_len), diagonal=1).bool()[:, :, None]
        output,total_decode_outputs,decode_attn_weights = self.decoder(self.drop(tgt_emb), self.drop(output),
            self.drop(tgt_pos_emb), dec_self_mask, enc_self_mask)
        output = self.out_layer(self.drop(output)) # ibv
        return output,total_encode_outputs,total_decode_outputs,decode_attn_weights

    def decode(self, src, enc_self_mask, eos_id, beam_size, max_len, hot_words=None, hot_words_scores=None):
        src_len, bsz = src.size(0), src.size(1)

        src_pos_seq = torch.arange(src_len - 1, -1, -1.0, device=src.device)
        if self.clamp_len > 0:
            src_pos_seq.clamp_(max=self.clamp_len)
        src_pos_emb = self.pos_emb(src_pos_seq)

        src_t = self.input_layer(src)

        enc_output,_ = self.encoder(self.drop(src_t), self.drop(src_pos_emb), enc_self_mask)
        # beam search algorithm
        # repeat the following steps
        # 1. get embeddings for beam_seqs
        # 2. throw embeddings and cache_states into decoder to obtain log_probs and new_states
        # 3. select top K top_tokens from log_probs, use beam_stops to mask log_probs
        # 4. use top_tokens to update beam_seqs, cache_states
        # 5. update beam_log_probs
        # 6. update beam_stops
        # return the first entry in beam_seqs

        device = src.device
        beam_seqs = torch.full((1, 1, bsz), eos_id, dtype=torch.long, device=device) # [beam, hlen, bsz]
        cache_k_states, cache_v_states = None, None # [beam, layer, hlen, bsz, d]
        beam_log_probs = torch.zeros(1, bsz, device=device) # [beam, bsz]
        beam_stops = torch.zeros(1, bsz, dtype=torch.uint8, device=device) # [beam, bsz]
        total_prob_seq = []
        if hot_words is not None:
            hot_words = torch.from_numpy(hot_words).to(device)
            hot_words_length = hot_words.shape[1] - (hot_words == -1).sum(1)
            hot_words_scores = torch.from_numpy(hot_words_scores).to(device)

            hw_match = torch.zeros(beam_size, bsz, hot_words.shape[0]).unsqueeze(-1).long().to(device)
            new_expect_tokens = None
            def score_func(score_scale):
                return torch.log10(1 / (0.8 * score_scale + 0.1))

        max_len = int(min(src_len * 0.8, max_len))
            
        tgt_pos_seq = torch.arange(max_len, -1, -1.0, device=device)
        if self.clamp_len > 0:
            tgt_pos_seq.clamp_(max=self.clamp_len)
        tgt_pos_emb = self.pos_emb(tgt_pos_seq)
        index_list, prefix_index_list = [], []
        for i in range(max_len):
            cur_emb = self.word_emb(beam_seqs[:, -1]) # [beam, bsz, emb]
            cur_emb.mul_(self.emb_scale)
            # output: [beam, bsz, d]
            # states: [beam, layer, bsz, d]
            if cache_k_states is None:
                k_input, v_input = None, None
            else:
                k_input, v_input = cache_k_states, cache_v_states
            output, new_k_states, new_v_states = self.decoder.decode_single_step(
                cur_emb, enc_output, k_input, v_input, tgt_pos_emb,
                enc_self_mask)
            logit = self.out_layer(output)
            log_probs = F.log_softmax(logit, -1)  # [beam, bsz, vocab]
            total_prob_seq.append(log_probs)
            if hot_words is not None:
                # get base score
                max_s, min_s = log_probs.max(dim=2)[0], log_probs.min(dim=2)[0]
                dif = max_s - min_s
                base_score = (dif * self.hw_alpha).unsqueeze(-1)

                hw_prefix = hot_words.transpose(1, 0)[None, :1].expand(log_probs.shape[0], bsz, -1)
                expect_list = torch.cat((hw_prefix, new_expect_tokens), dim=2) if hw_match.sum() else hw_prefix
                raw_scores = log_probs.gather(2, expect_list)
                score_scale = (raw_scores - min_s.unsqueeze(-1)) / dif.unsqueeze(-1)
                scores = base_score * score_func(score_scale)

                if hw_match.sum():
                    hw_mask = (hw_match > 0).squeeze(-1).float() * 1.5
                    hw_mask[hw_mask == 0] = 1.0
                    expect_mask = torch.cat((torch.ones_like(hw_mask), hw_mask), dim=2)

                    # sort to avoid score overwriting
                    expect_mask, mask_idx = torch.sort(expect_mask, dim=2)
                    raw_scores = raw_scores.gather(dim=2, index=mask_idx)
                    scores = scores.gather(dim=2, index=mask_idx)
                    expect_list = expect_list.gather(dim=2, index=mask_idx)
                    
                    scores = scores * expect_mask

                log_probs.scatter_(2, expect_list, raw_scores + scores)
            
            # pick only the first for eos beams
            beam_stops_flt = beam_stops.float()
            log_probs[:, :, 0] *= (1 - beam_stops_flt)
            log_probs[:, :, 1:] -= beam_stops_flt[:, :, None] * 1e30

            log_probs += beam_log_probs[:, :, None]

            nvocab = log_probs.size(2)
            log_probs = log_probs.permute(0, 2, 1).contiguous().view(-1, bsz) # [beam x vocab, bsz]
            beam_log_probs, top_tokens = torch.topk(log_probs, beam_size, 0) # [beam, bsz]
            top_beam_idx = (top_tokens // nvocab).long() # [beam, bsz]
            top_vocab_idx = top_tokens.float().fmod(nvocab).long() # [beam, bsz]

            t_top_beam_idx = top_beam_idx[:, None].expand(-1, beam_seqs.size(1), -1) # [beam, hlen, bsz]
            beam_seqs = torch.gather(beam_seqs, 0, t_top_beam_idx) # [beam, hlen, bsz]
            beam_seqs = torch.cat([beam_seqs, top_vocab_idx[:, None]], 1)
            if hot_words is not None:
                # order hw_match tensor according to beam_idx
                b_idx = top_beam_idx.unsqueeze(-1).expand(-1, -1, hot_words.size()[0])
                hw_match = hw_match.squeeze(-1).gather(0, b_idx).unsqueeze(-1)

                # expect hw token
                hw_expand = hot_words[None, None, :, :].expand(beam_size, bsz, -1, -1).gather(3, hw_match)
                # coming token of pred
                seq_expand = beam_seqs.permute(0, 2, 1).contiguous()[:, :, None, -1:].expand(-1, -1, hot_words.size()[0], -1)

                # update match
                match = (hw_expand == seq_expand)
                expand_match = hw_match.masked_select(match) + 1
                hw_match = torch.zeros_like(hw_match).masked_scatter(match, expand_match)

                # handle with the situation when the whole hw is matched
                tot_len = hot_words_length[None, None, :].expand(beam_size, bsz, -1)
                complete_match = (hw_match.squeeze(-1) == tot_len).unsqueeze(-1)
                hw_match.masked_fill_(complete_match, 0)

                # get new expecting token id
                new_expect_tokens = hot_words[None, None, :, :].expand(beam_size, bsz, -1, -1).gather(3, hw_match).squeeze(-1)
            k_input = new_k_states[:, :, None]
            v_input = new_v_states[:, :, None]
            if cache_k_states is None:
                cache_k_states, cache_v_states = k_input, v_input
            else:
                cache_k_states = torch.cat([cache_k_states, k_input], 2)
                cache_v_states = torch.cat([cache_v_states, v_input], 2)
            c_size = cache_k_states.size()
            t_top_beam_idx = top_beam_idx[:, None, None, :, None].expand(-1, c_size[1], c_size[2], -1, c_size[4])
            cache_k_states = torch.gather(cache_k_states, 0, t_top_beam_idx) # [beam, layer, hlen, bsz, d]
            cache_v_states = torch.gather(cache_v_states, 0, t_top_beam_idx) # [beam, layer, hlen, bsz, d]

            beam_stops = torch.gather(beam_stops, 0, top_beam_idx)
            beam_stops = beam_stops | (top_vocab_idx.long() == eos_id)

            # if beam_stops[0].all().item():
                # break
        # return beam_seqs[0][1:], beam_log_probs[0],torch.stack(total_prob_seq).squeeze() # [hlen, bsz], remove the first eos
        return beam_seqs[0][1:], beam_log_probs[0],None # [hlen, bsz], remove the first eos