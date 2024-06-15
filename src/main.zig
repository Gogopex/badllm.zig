const std = @import("std");
const posix = std.posix;

const NUM_ACTIVATION_TENSORS = 23;

const FileOpenError = error{
    InvalidHeader,
    UnsupportedVersion,
    FileEmpty,
    FileTooSmall,
};

const ModelError = error{
    ModelNotInitialized,
    BatchSizeTooLarge,
};

const GPT2Config = struct {
    max_seq_len: u32, // Maximum sequence length eg 1024
    vocab_size: u32, // Vocabulary size eg 50257
    n_embed: u32, // Embedding dimension eg 768
    n_layer: u32, // Number of layers eg 12
    n_head: u32, // Number of attention heads eg 12
    n_channels: u32, // Number of channels in the MLP eg 768
    padded_vocab_size: usize,
};

const GPT2 = struct {
    config: GPT2Config,
    num_parameters: u32,
    params: []f32,
    params_memory: []f32,
    params_size: [16]usize,
    activations: []f32,
    activations_memory: []f32,
    activation_sizes: [23]usize,
    gradients: []f32,
    gradients_memory: []f32,
    gradients_activations: []f32,
    gradients_activations_memory: []f32,
    // AdamW optimizer buffers
    m: []f32,
    m_memory: []f32,
    v: []f32,
    v_memory: []f32,
    inputs: ?[]u8,
    targets: ?[]u8,
    batch_size: usize = 0,
    seq_len: usize = 0,
    mean_loss: f32 = -1.0, // TODO ?

    var default = GPT2{
        .config = GPT2Config{
            .max_seq_len = 0,
            .vocab_size = 0,
            .padded_vocab_size = 0,
            .n_layer = 0,
            .n_embed = 0,
            .n_head = 0,
            .n_channels = 0,
        },
        .num_parameters = 0,
        .params = &[_]f32{},
        .params_memory = &[_]f32{},
        .params_size = [_]usize{0} ** 16,
        .activations = &[_]f32{},
        .activations_memory = &[_]f32{},
        .activation_sizes = [_]usize{0} ** 23, // explain to non-ziglets
        .gradients = &[_]f32{},
        .gradients_memory = &[_]f32{},
        .gradients_activations = &[_]f32{},
        .gradients_activations_memory = &[_]f32{},
        .m = &[_]f32{},
        .m_memory = &[_]f32{},
        .v = &[_]f32{},
        .v_memory = &[_]f32{},
        .inputs = null,
        .targets = null,
        .batch_size = 0,
        .seq_len = 0,
        .mean_loss = -1.0,
    };
};

//
// const GPT = struct {
//     config: GPT2Config,
//     // weights: ParameterTensors,
// };
//
// // TODO: list all all acronyms here instead
// // link to multiheaded attention, wte, wpe, layer norm, attention core concepts
const ParameterTensors = struct {
    word_token_embeddings: f32, // shape V, C where V is vocab size, C is embedding dims -- each word in the vocab is mapped to a vector of size C
    word_position_embeddings: f32, // shape maxT, C -- maxT is maximum sequence length, C is embeddingdims -- adds positional info to the token embeddings
    layer_norm_weights_layer_1: f32, // shape L, C -- L is the num of layers, C embedding dims
    layer_norm_biases_layer_1: f32, // shape L, C -- L is the num of layers, C embedding dims
    qkvw: f32, // shape L, 3C, C -- query key values weight projections for multiheaded attention -- L is num of layers, 3C is query/key/values concat, C is the embedding dims
    qkvb: f32, // shape L, 3C -- query key values bias projections for multiheaded attention
    attention_projection_weights: f32, // shape L, C, C -- weights of the concat output of the attention heads back to the embedding dimension
    attention_projection_biases: f32, // shape L, C -- biases of the concat output of the attention heads back to the embedding dimension
    layer_norm_weights_layer_2: f32, //
    layer_norm_biases_layer_2: f32,
    feed_forward_weights: f32, // shape L, 4C, C -- weights of the FFN
    feed_forward_biases: f32, // shape L, 4C -- biases of the FFN
    feed_forward_projection_weights: f32, // L, C, 4C -- weights for projecting the output of the FFN back to the embedding dimension
    final_layer_norm_weights: f32, // shape C -- final weights for the final layer norm
    final_layer_norm_biases: f32, // shape C -- final biases for the final layer norm
};

const ActivationTensors = struct {
    encoded: []f32, // (B, T, C)
    ln1: []f32, // (L, B, T, C)
    ln1_mean: []f32, // (L, B, T)
    ln1_rstd: []f32, // (L, B, T)
    qkv: []f32, // (L, B, T, 3*C)
    atty: []f32, // (L, B, T, C)
    preatt: []f32, // (L, B, NH, T, T)
    att: []f32, // (L, B, NH, T, T)
    attproj: []f32, // (L, B, T, C)
    residual2: []f32, // (L, B, T, C)
    ln2: []f32, // (L, B, T, C)
    ln2_mean: []f32, // (L, B, T)
    ln2_rstd: []f32, // (L, B, T)
    fch: []f32, // (L, B, T, 4*C)
    fch_gelu: []f32, // (L, B, T, 4*C)
    fcproj: []f32, // (L, B, T, C)
    residual3: []f32, // (L, B, T, C)
    lnf: []f32, // (B, T, C)
    lnf_mean: []f32, // (B, T)
    lnf_rstd: []f32, // (B, T)
    logits: []f32, // (B, T, V)
    probs: []f32, // (B, T, V)
    losses: []f32, // (B, T)
};
pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var model = GPT2.default;
    build_model_from_file("gpt2_124M.bin", &model) catch |err| {
        std.debug.print("Error building model: {}\n", .{err});
        return;
    };

    // TODO: write-up on why 4 and 64
    var train_loader: DataLoader = undefined;
    const B = 4;
    const T = 64; // TODO: store in struct init directly?
    try dataloader_init(allocator, &train_loader, "data/tiny_shakespeare_train.bin", B, T);
    std.debug.print("train dataset num_batches: {}\n", .{train_loader.num_batches});

    var val_loader: DataLoader = undefined;
    try dataloader_init(allocator, &val_loader, "data/tiny_shakespeare_val.bin", B, T);
    std.debug.print("val dataset num_batches: {}\n", .{val_loader.num_batches});

    const val_num_batches = 10;
    const rng_state = 1337;
    const gen_max_length = 64;
    var gen_tokens: [gen_max_length]u32 = undefined;

    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.os.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();

    var ts: std.posix.timespec = undefined;
    for (0..40) |step| {
        if (step % 10 == 0) {
            var val_loss: f32 = 0.0;
            data_loader_reset(&val_loader);
            for (0..val_num_batches) |_| {
                data_loader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            std.debug.print("val loss: {}\n", .{val_loss});
        }
        if (step > 0 and step % 20 == 0) {
            gen_tokens[0] = 50256;

            for (1..gen_max_length) |t| {
                gpt2_forward(&model, gen_tokens, null, 1, t);
                const probs = model.acts.probs + (t - 1) * model.config.vocab_size;
                const coin = rand.float(f32);
                const next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
            }
            std.debug.print("generated: ", .{});
            for (0..gen_max_length) |t| {
                std.debug.print("{} ", .{gen_tokens[t]});
            }
            std.debug.print("\n", .{});
        }
        posix.times(&ts);
        data_loader_next_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        // gpt2_zero_grad(&model);
        // gpt2_backward(&model);
        // gpt2_update(&model, 1e-4, 0.9, 0.999, 1e-8, 0.0, step + 1);
        posix.times(&ts);
        const time_elapsed_s = (ts.tv_sec - ts.tv_sec) + (ts.tv_nsec - ts.tv_nsec) / 1e9;
        std.debug.print("step {}: train loss {} (took {} ms)\n", .{ step, model.mean_loss, time_elapsed_s * 1000 });
    }

    // some memory for generating samples from the model
    // unsigned long long rng_state = 1337;
    // const int gen_max_length = 64;
    // int gen_tokens[gen_max_length];
    //
    // // train
    // struct timespec start, end;
    // for (int step = 0; step <= 40; step++) {
    //
    //     // once in a while estimate the validation loss
    //     if (step % 10 == 0) {
    //         float val_loss = 0.0f;
    //         dataloader_reset(&val_loader);
    //         for (int i = 0; i < val_num_batches; i++) {
    //             dataloader_next_batch(&val_loader);
    //             gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
    //             val_loss += model.mean_loss;
    //         }
    //         val_loss /= val_num_batches;
    //         printf("val loss %f\n", val_loss);
    //     }
    //
    //     // once in a while do model inference to print generated text
    //     if (step > 0 && step % 20 == 0) {
    //         gen_tokens[0] = GPT2_EOT; // the GPT-2 EOT token kicks off the generation
    //         for (int t = 1; t < gen_max_length; t++) {
    //             // note that inference is wasteful here because
    //             // for each t, we re-compute all activations between 0 and t
    //             // leaving this alone because you want separate code for inference anyway
    //             // the inference here is just for sanity checking purposes
    //             gpt2_forward(&model, gen_tokens, NULL, 1, t);
    //             float* probs = model.acts.probs + (t-1) * model.config.vocab_size;
    //             float coin = random_f32(&rng_state);
    //             int next_token = sample_mult(probs, model.config.vocab_size, coin);
    //             gen_tokens[t] = next_token;
    //         }
    //         printf("generated: ");
    //         for (int t = 0; t < gen_max_length; t++) {
    //             printf("%d ", gen_tokens[t]);
    //         }
    //         printf("\n");
    //     }
    //
    //     // do a training step
    //     clock_gettime(CLOCK_MONOTONIC, &start);
    //     dataloader_next_batch(&train_loader);
    //     gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
    //     gpt2_zero_grad(&model);
    //     gpt2_backward(&model);
    //     gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
    //     clock_gettime(CLOCK_MONOTONIC, &end);
    //     double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    //     printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
}

// int sample_mult(float* probabilities, int n, float coin) {
//     // sample index from probabilities (they must sum to 1!)
//     // coin is a random number in [0, 1), usually from random_f32()
//     float cdf = 0.0f;
//     for (int i = 0; i < n; i++) {
//         cdf += probabilities[i];
//         if (coin < cdf) {
//             return i;
//         }
//     }
//     return n - 1; // in case of rounding errors
// }

fn sample_mult(probabilities: f32, n: u32, coin: f32) void {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()

    const cdf = 0.0;

    for (0..n) |i| {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors

}

pub fn gpt2_forward(allocator: std.mem.Allocator, model: GPT2, inputs: []u32, targets: []u32, B: u32, T: u32) !void {
    if (model.params_memory == null) {
        std.debug.print("Error: model was not initialized properly.\n", .{});

        return ModelError.ModelNotInitialized;
    }

    // convenience parameters
    const V = model.config.vocab_size;
    const L = model.config.n_layer;
    const NH = model.config.n_head;
    const C = model.config.n_channels;

    if (model.activations_memory == null) {
        model.seq_len = T;
        model.act_sizes[0] = B * T * C;
        model.act_sizes[1] = L * B * T * C;
        model.act_sizes[2] = L * B * T;
        model.act_sizes[3] = L * B * T;
        model.act_sizes[4] = L * B * T * 3 * C;
        model.act_sizes[5] = L * B * T * C;
        model.act_sizes[6] = L * B * NH * T * T;
        model.act_sizes[7] = L * B * NH * T * T;
        model.act_sizes[8] = L * B * T * C;
        model.act_sizes[9] = L * B * T * C;
        model.act_sizes[10] = L * B * T * C;
        model.act_sizes[11] = L * B * T;
        model.act_sizes[12] = L * B * T;
        model.act_sizes[13] = L * B * T * 4 * C;
        model.act_sizes[14] = L * B * T * 4 * C;
        model.act_sizes[15] = L * B * T * C;
        model.act_sizes[16] = L * B * T * C;
        model.act_sizes[17] = B * T * C;
        model.act_sizes[18] = B * T;
        model.act_sizes[19] = B * T;
        model.act_sizes[20] = B * T * V;
        model.act_sizes[21] = B * T * V;
        model.act_sizes[22] = B * T;

        const num_activations = 0;

        for (0..NUM_ACTIVATION_TENSORS) |i| {
            num_activations += model.act_sizes[i];
        }
        std.debug.print("num_activations: {}", .{num_activations});
        model.num_activations = num_activations;
        model.activations_memory = try allocator.alloc(f32, num_activations);
        model.inputs = try allocator.alloc(u32, B * T);
        model.targets = try allocator.alloc(u32, B * T);
    } else {
        if (B > model.batch_size or T > model.seq_len) {
            std.debug.print("Error: batch size or sequence length is inadequately large\n", .{});
            std.debug.print("Model: B={} T={}, Desired: B={} T={}\n", .{ model.batch_size, model.seq_len, B, T });
            return ModelError.BatchSizeTooLarge;
        }
    }
    std.mem.copy(model.inputs, inputs, B * T * @sizeOf(u32));
    if (targets != null) {
        std.mem.copy(model.targets, targets, B * T * @sizeOf(u32));
    }

    const params: ParameterTensors = model.params; // ???
    const acts: ActivationTensors = model.activations; // ???
    const residual: f32 = undefined;
    for (0..L) |l| {
        residual = if (1 == 0) acts.encoded else acts.residual3 + (l-1) * B * T * C;


        // weights for this layer
        const l_ln1w = params.layer_norm_weights_layer_1[l * C..];
        const l_ln1b = params.layer_norm_biases_layer_1[l * C..];
        const l_qkvw = params.qkvw[l * 3*C * C];
        const l_qkvb = params.qkvb[l * 3*C];
        const l_attprojw = params.attention_projection_weights[l * C * C];
        const l_attprojb = params.attention_projection_biases[l * C];

        // activations for this layer
        // TODO: continue here
    }
    //  for (int l = 0; l < L; l++) {
    //
    //     residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
    //
    //     // get the pointers of the weights for this layer
    //     float* l_ln1w = params.ln1w + l * C;
    //     float* l_ln1b = params.ln1b + l * C;
    //     float* l_qkvw = params.qkvw + l * 3*C * C;
    //     float* l_qkvb = params.qkvb + l * 3*C;
    //     float* l_attprojw = params.attprojw + l * C * C;
    //     float* l_attprojb = params.attprojb + l * C;
    //     float* l_ln2w = params.ln2w + l * C;
    //     float* l_ln2b = params.ln2b + l * C;
    //     float* l_fcw = params.fcw + l * 4*C * C;
    //     float* l_fcb = params.fcb + l * 4*C;
    //     float* l_fcprojw = params.fcprojw + l * C * 4*C;
    //     float* l_fcprojb = params.fcprojb + l * C;
    //
    //     // get the pointers of the activations for this layer
    //     float* l_ln1 = acts.ln1 + l * B * T * C;
    //     float* l_ln1_mean = acts.ln1_mean + l * B * T;
    //     float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
    //     float* l_qkv = acts.qkv + l * B * T * 3*C;
    //     float* l_atty = acts.atty + l * B * T * C;
    //     float* l_preatt = acts.preatt + l * B * NH * T * T;
    //     float* l_att = acts.att + l * B * NH * T * T;
    //     float* l_attproj = acts.attproj + l * B * T * C;
    //     float* l_residual2 = acts.residual2 + l * B * T * C;
    //     float* l_ln2 = acts.ln2 + l * B * T * C;
    //     float* l_ln2_mean = acts.ln2_mean + l * B * T;
    //     float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
    //     float* l_fch = acts.fch + l * B * T * 4*C;
    //     float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
    //     float* l_fcproj = acts.fcproj + l * B * T * C;
    //     float* l_residual3 = acts.residual3 + l * B * T * C;
    //
    //     // now do the forward pass
    //     layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
    //     matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
    //     attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
    //     matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
    //     residual_forward(l_residual2, residual, l_attproj, B*T*C);
    //     layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
    //     matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
    //     gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
    //     matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
    //     residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    // }
    // residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    // layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    // matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, V);
    // softmax_forward(acts.probs, acts.logits, B, T, V);
    //
    // // also forward the cross-entropy loss function if we have the targets
    // if (targets != NULL) {
    //     crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, V);
    //     // for convenience also evaluate the mean loss
    //     float mean_loss = 0.0f;
    //     for (int i=0; i<B*T; i++) { mean_loss += model->acts.losses[i]; }
    //     mean_loss /= B*T;
    //     model->mean_loss = mean_loss;
    // } else {
    //     // if we don't have targets, we don't have a loss
    //     model->mean_loss = -1.0f;
    // }
}

// void encoder_forward(float* out,
//                    int* inp, float* wte, float* wpe,
//                    int B, int T, int C) {
//     for (int b = 0; b < B; b++) {
//         for (int t = 0; t < T; t++) {
//             // seek to the output position in out[b,t,:]
//             float* out_bt = out + b * T * C + t * C;
//             // get the index of the token at inp[b, t]
//             int ix = inp[b * T + t];
//             // seek to the position in wte corresponding to the token
//             float* wte_ix = wte + ix * C;
//             // seek to the position in wpe corresponding to the position
//             float* wpe_t = wpe + t * C;
//             // add the two vectors and store the result in out[b,t,:]
//             for (int i = 0; i < C; i++) {
//                 out_bt[i] = wte_ix[i] + wpe_t[i];
//             }
//         }
//     }
// }

fn encoder_forward(out: []f32, inp: []u32, wte: []f32, wpe: []f32, B: u32, T: u32, C: u32) void {
    for (0..B) |b| {
        for (0..T) |t| {
            var out_bt = out + b * T * C + t * C;
            var ix = inp[b * T + t];
            var wte_ix = wte + ix * C;
            var wpe_t = wpe + t * C;
            for (0..C) |i| {
                out_bt[i] = wte_ix[i] + wpe_t[i];
        }
    }
}

// pub fn gelu(x: f32) f32 {
//     return 0.5 * x * (1.0 + std.math.tan(std.math.sqrt(2.0 / std.math.pi) * (x + 0.044715 * pow(x, 3.9))));
// }

// fn layer_norm(x: []f32, eps: f32) []32 {
//     const mean = std.math.mean(x);
//     const variance = std.math.variance(x, mean);
//     const normalized = std.heap.page_allocator.create([]f32, x.len) catch unreachable;
//
//     for (x, 0..) |value, i| {
//         normalized[i] = (value - mean) / std.math.sqrt(variance + eps);
//     }
//
//     return normalized;
// }

const Tokenizer = struct { vocabulary_size: u32, init: bool, data: []u8 };
const DataLoader = struct {
    B: u32,
    T: u32,
    tokens_file: std.fs.File,
    file_size: u64,
    current_position: u32,
    batch: []u32,
    inputs: []u32,
    targets: []u32,
    num_batches: u64,
};

fn build_model_from_file(filepath: []const u8, model: *GPT2) !void {
    const t = try read_n_from_checkpoint_file(filepath, 256, u32, 0);
    const headers = t.items;

    const config = GPT2Config{
        .max_seq_len = headers[2],
        .vocab_size = headers[3],
        .n_embed = 0.0,
        .n_layer = headers[4],
        .n_head = headers[5],
        .n_channels = headers[6],
        .padded_vocab_size = 0,
    };

    // TODO: breakdown and explanation for newbies
    // at this point I might as well turn the repo into a full-on breakdown of decoder-only transformers
    model.params_size[0] = config.vocab_size * config.max_seq_len;
    model.params_size[1] = config.max_seq_len * config.n_channels;
    model.params_size[2] = config.n_layer * config.n_channels;
    model.params_size[3] = config.n_layer * config.n_channels;
    model.params_size[4] = config.n_layer * (3 * config.n_channels) * config.n_channels;
    model.params_size[5] = config.n_layer * (3 * config.n_channels);
    model.params_size[6] = config.n_layer * config.n_channels * config.n_channels;
    model.params_size[7] = config.n_layer * config.n_channels;
    model.params_size[8] = config.n_layer * config.n_channels;
    model.params_size[9] = config.n_layer * config.n_channels;
    model.params_size[10] = config.n_layer * (4 * config.n_channels) * config.n_channels;
    model.params_size[11] = config.n_layer * (4 * config.n_channels);
    model.params_size[12] = config.n_layer * config.n_channels * (4 * config.n_channels);
    model.params_size[13] = config.n_layer * config.n_channels;
    model.params_size[14] = config.n_channels;
    model.params_size[15] = config.n_channels;

    std.debug.print("{}, {}\n", .{ model.params_size[7], model.params_size[12] });
    std.debug.print("max_seq_len: {}, vocab_size: {}, n_layer: {}, n_head: {}, n_channels: {}\n", .{ config.max_seq_len, config.vocab_size, config.n_layer, config.n_head, config.n_channels });

    var num_parameters: u32 = 0;
    for (0..16) |i| {
        num_parameters += @intCast(model.params_size[i]);
    }

    model.num_parameters = num_parameters;

    std.debug.print("num parameters: {}\n", .{num_parameters});

    const params_memory: std.ArrayList(f32) = try read_n_from_checkpoint_file(filepath, model.num_parameters, f32, 256);
    model.params_memory = params_memory.items;

    std.debug.print("params_memory[1]: {}\n", .{model.params_memory[1]});

    // // read in all the parameters from file
    // model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    // fread(model->params_memory, sizeof(float), num_parameters, model_file);
    // fclose(model_file);
    //
    // // other inits
    // model->acts_memory = NULL;
    // model->grads_memory = NULL;
    // model->m_memory = NULL;
    // model->v_memory = NULL;
    // model->grads_acts_memory = NULL;
    // model->inputs = NULL;
    // model->targets = NULL;
    // model->batch_size = 0;
    // model->seq_len = 0;
    // model->mean_loss = -1.0f; // -1.0f will designate no loss
    model.config = config;
}

fn build_tokenizer_from_vocab(filepath: []const u8, tokenizer: *Tokenizer) !void {
    const headers = try read_n_from_checkpoint_file(filepath, 256, 0) catch |err| {
        std.debug.print("Error reading the tokenizer file\n", .{});
        return err;
    };

    if (headers[0] != 20240326) {
        std.debug.print("Bad magic tokenizer file\n", .{});
        return FileOpenError.InvalidHeader;
    }

    if (headers[1] != 1) {
        std.debug.print("Bad version in tokenizer file\n", .{});
        return FileOpenError.UnsupportedVersion;
    }

    var file = try std.fs.cwd().openFile(filepath, .{ .mode = .read_only }) catch |err| {
        std.debug.print("Error opening the tokenizer file\n", .{});
        return err;
    };

    defer file.close();

    try file.seekTo(256 * @sizeOf(f32));

    tokenizer.vocab_size = headers[2];
    tokenizer.data = std.heap.page_allocator.create([]u8, tokenizer.vocab_size * f32) catch unreachable;

    for (tokenizer.vocab_map) |*token| {
        var token_length: u8 = undefined;
        try file.readAll(std.mem.asBytes(&token_length));
        token.* = try std.heap.page_allocator.create([]u8, token_length) catch unreachable;
        try file.readAll(token.*);
    }

    std.debug.print("sample token", .{tokenizer.vocab_map[69]});

    tokenizer.init = true;
}

// need both f32 and u32 handling for GPT2Config vs parameters
fn read_n_from_checkpoint_file(filepath: []const u8, N: usize, comptime T: type, offset: usize) !std.ArrayList(T) {
    var file = try std.fs.cwd().openFile(filepath, .{ .mode = .read_only });
    defer file.close();

    const file_size = try file.getEndPos();
    std.debug.print("{}, {}, {} \n", .{ file_size, N, @sizeOf(T) });
    if (file_size == 0) {
        return error.FileEmpty;
    }

    try file.seekTo(offset * @sizeOf(T));

    var data = try std.ArrayList(T).initCapacity(std.heap.page_allocator, N);
    try std.ArrayList(T).resize(&data, N);

    const bytes = std.mem.sliceAsBytes(data.items);
    _ = try file.read(bytes);

    return data;
}

fn dataloader_init(allocator: std.mem.Allocator, loader: *DataLoader, filepath: []const u8, B: u32, T: u32) !void {
    loader.B = B;
    loader.T = T;

    var file = try std.fs.cwd().openFile(filepath, .{ .mode = .read_only });
    defer file.close();

    const file_size = try file.getEndPos();
    std.debug.print("{} \n", .{file_size});
    if (file_size == 0) {
        return error.FileEmpty;
    }
    loader.file_size = file_size;
    loader.current_position = 0;
    std.debug.print("B, T: {}, {}", .{ B, T });
    loader.batch = try allocator.alloc(u32, (B * T + 1) * @sizeOf(u32));
    loader.inputs = loader.batch;
    loader.targets = loader.batch[1..];
    loader.num_batches = file_size / (B * T * @sizeOf(u32));
}

fn data_loader_reset(loader: *DataLoader) void {
    loader.current_position = 0;
}

fn data_loader_next_batch(loader: *DataLoader) void {
    const B = loader.B;
    const T = loader.T;
    if (loader.current_position + (B * T + 1) * @sizeOf(u32) > loader.file_size) {
        loader.current_position = 0;
    }
    try loader.tokens_file.seekTo(loader.current_position);
    try loader.tokens_file.readAll(std.mem.sliceAsBytes(loader.batch));
    loader.current_position += B * T * @sizeOf(u32);
}

// fn load_embeddings(filepath: []const u8, vocab_size: usize, embedding_dim: usize) []f32 {
//     return load_weights(filepath, vocab_size * embedding_dim);
// }
//
// fn load_positional_encodings(filepath: []const u8, seq_len: usize, embedding_dim: usize) []f32 {
//     return load_weights(filepath, seq_len * embedding_dim);
// }

// TODO: design basic forward pass implementation
// load weights and embeddings
// multiheaded attention
// FFN
// add & norm

test "simple test" {
    try std.testing.expectEqual(1, 1);
}
