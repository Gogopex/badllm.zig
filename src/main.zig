const std = @import("std");
const builtin = @import("builtin");
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
    params: ParameterTensors,
    params_memory: []f32,
    params_size: [16]usize,
    activations: ActivationTensors,
    activations_memory: []f32,
    num_activations: u32,
    activation_sizes: [23]u32,
    gradients: []f32,
    gradients_memory: []f32,
    gradients_activations: []f32,
    gradients_activations_memory: []f32,
    // AdamW optimizer buffers
    m: []f32,
    m_memory: []f32,
    v: []f32,
    v_memory: []f32,
    inputs: []u32,
    targets: []u32,
    batch_size: usize = 0,
    seq_len: usize = 0,
    mean_loss: f32 = -1.0, // TODO ?
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
    encoded: []f32,
    word_token_embeddings: []f32, // shape V, C where V is vocab size, C is embedding dims -- each word in the vocab is mapped to a vector of size C
    word_position_embeddings: []f32, // shape maxT, C -- maxT is maximum sequence length, C is embeddingdims -- adds positional info to the token embeddings
    layer_norm_weights_layer_1: []f32, // shape L, C -- L is the num of layers, C embedding dims
    layer_norm_biases_layer_1: []f32, // shape L, C -- L is the num of layers, C embedding dims
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
    ln1: []f32, // (L, B, T, C), layer norm 1
    ln1_mean: []f32, // (L, B, T), layer norm 1 mean
    ln1_rstd: []f32, // (L, B, T), layer norm 1 r
    qkv: []f32, // (L, B, T, 3*C), reciprocal of standard deviation
    atty: []f32, // (L, B, T, C), attention output
    preatt: []f32, // (L, B, NH, T, T), pre-attention
    att: []f32, // (L, B, NH, T, T), attention
    attproj: []f32, // (L, B, T, C), attention projection
    residual2: []f32, // (L, B, T, C), residual 2
    ln2: []f32, // (L, B, T, C), layer norm 2
    ln2_mean: []f32, // (L, B, T), layer norm 2 mean
    ln2_rstd: []f32, // (L, B, T), layer norm 2 reciprocal of standard deviation
    fch: []f32, // (L, B, T, 4*C), fully connected hidden
    fch_gelu: []f32, // (L, B, T, 4*C), fully connected hidden gelu
    fcproj: []f32, // (L, B, T, C), fully connected projection
    residual3: []f32, // (L, B, T, C), residual 3
    lnf: []f32, // (B, T, C), layer norm final
    lnf_mean: []f32, // (B, T), layer norm final mean
    lnf_rstd: []f32, // (B, T), layer norm final reciprocal of standard deviation
    logits: []f32, // (B, T, V)
    probs: []f32, // (B, T, V)
    losses: []f32, // (B, T)
};
pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var model: GPT2 = undefined;
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

    std.debug.print("train_loader file: {}", .{train_loader.tokens_file});
    var val_loader: DataLoader = undefined;
    try dataloader_init(allocator, &val_loader, "data/tiny_shakespeare_val.bin", B, T);
    std.debug.print("val dataset num_batches: {}\n", .{val_loader.num_batches});

    const val_num_batches = 10;
    // const rng_state = 1337;
    const gen_max_length = 64;
    var gen_tokens: [gen_max_length]u32 = undefined;

    const state = 1337;
    var prng = std.rand.DefaultPrng.init(state);
    const rand = prng.random();

    for (0..40) |step| {
        if (step % 10 == 0) {
            var val_loss: f32 = 0.0;
            data_loader_reset(&val_loader);
            std.debug.print("val_loader: {}", .{val_loader.file_size});
            for (0..val_num_batches) |_| {
                try data_loader_next_batch(&val_loader);
                try gpt2_forward(allocator, &model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            std.debug.print("Val loss: {}\n", .{val_loss});
        }
        if (step > 0 and step % 20 == 0) {
            gen_tokens[0] = 50256;

            for (1..gen_max_length) |t| {
                const no_targs: []u32 = undefined; // @todo maybe make targets ? in gpt2_forward and switch on null
                try gpt2_forward(allocator, &model, &gen_tokens, no_targs, 1, @intCast(t));
                const probs = model.activations.probs[(t - 1) * model.config.vocab_size ..];
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
        try data_loader_next_batch(&train_loader);
        try gpt2_forward(allocator, &model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        // gpt2_backward(&model);
        // gpt2_update(&model, 1e-4, 0.9, 0.999, 1e-8, 0.0, step + 1);
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

fn sample_mult(probabilities: []f32, n: u32, coin: f32) u32 {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()

    var cdf: f32 = 0.0;

    for (0..n) |i| {
        cdf += probabilities[i];
        if (coin < cdf) {
            return @as(u32, @intCast(i));
        }
    }
    return n - 1; // in case of rounding errors

}

fn gpt2_zero_grad(model: *GPT2) void {
    if (model.gradients_memory.len != 0) {
        @memset(model.gradients_memory, 0);
    }
    if (model.gradients_activations_memory.len != 0) {
        @memset(model.grads_acts_memory, 0);
    }
}

pub fn gpt2_forward(allocator: std.mem.Allocator, model: *GPT2, inputs: []u32, targets: []u32, B: u32, T: u32) !void {
    // @TODO: handle eventual null
    if (model.params_memory.len == 0) {
        std.debug.print("Error: model was not initialized properly.\n", .{});

        return ModelError.ModelNotInitialized;
    }

    std.debug.print("gpt2_forward\n", .{});

    // convenience parameters
    const V = model.config.vocab_size;
    const L = model.config.n_layer;
    const NH = model.config.n_head;
    const C = model.config.n_channels;

    if (model.activations_memory.len == 0) {
        model.batch_size = B;
        model.seq_len = T;
        model.activation_sizes[0] = B * T * C;
        model.activation_sizes[1] = L * B * T * C;
        model.activation_sizes[2] = L * B * T;
        model.activation_sizes[3] = L * B * T;
        model.activation_sizes[4] = L * B * T * 3 * C;
        model.activation_sizes[5] = L * B * T * C;
        model.activation_sizes[6] = L * B * NH * T * T;
        model.activation_sizes[7] = L * B * NH * T * T;
        model.activation_sizes[8] = L * B * T * C;
        model.activation_sizes[9] = L * B * T * C;
        model.activation_sizes[10] = L * B * T * C;
        model.activation_sizes[11] = L * B * T;
        model.activation_sizes[12] = L * B * T;
        model.activation_sizes[13] = L * B * T * 4 * C;
        model.activation_sizes[14] = L * B * T * 4 * C;
        model.activation_sizes[15] = L * B * T * C;
        model.activation_sizes[16] = L * B * T * C;
        model.activation_sizes[17] = B * T * C;
        model.activation_sizes[18] = B * T;
        model.activation_sizes[19] = B * T;
        model.activation_sizes[20] = B * T * V;
        model.activation_sizes[21] = B * T * V;
        model.activation_sizes[22] = B * T;
        // Lets compute the number of activations. Mostly for debugging reasons.
        var num_activations: u32 = 0;
        for (model.activation_sizes) |size| {
            num_activations += @as(u32, @intCast(size));
        }
        model.num_activations = num_activations;
        model.activations_memory = try allocator.alloc(f32, num_activations);
        var iter: u32 = 0;
        model.activations.encoded = model.activations_memory[iter .. iter + model.activation_sizes[0]];
        iter += model.activation_sizes[0];
        model.activations.ln1 = model.activations_memory[iter .. iter + model.activation_sizes[1]];
        iter += model.activation_sizes[1];
        model.activations.ln1_mean = model.activations_memory[iter .. iter + model.activation_sizes[2]];
        iter += model.activation_sizes[2];
        model.activations.ln1_rstd = model.activations_memory[iter .. iter + model.activation_sizes[3]];
        iter += model.activation_sizes[3];
        model.activations.qkv = model.activations_memory[iter .. iter + model.activation_sizes[4]];
        iter += model.activation_sizes[4];
        model.activations.atty = model.activations_memory[iter .. iter + model.activation_sizes[5]];
        iter += model.activation_sizes[5];
        model.activations.preatt = model.activations_memory[iter .. iter + model.activation_sizes[6]];
        iter += model.activation_sizes[6];
        model.activations.att = model.activations_memory[iter .. iter + model.activation_sizes[7]];
        iter += model.activation_sizes[7];
        model.activations.attproj = model.activations_memory[iter .. iter + model.activation_sizes[8]];
        iter += model.activation_sizes[8];
        model.activations.residual2 = model.activations_memory[iter .. iter + model.activation_sizes[9]];
        iter += model.activation_sizes[9];
        model.activations.ln2 = model.activations_memory[iter .. iter + model.activation_sizes[10]];
        iter += model.activation_sizes[10];
        model.activations.ln2_mean = model.activations_memory[iter .. iter + model.activation_sizes[11]];
        iter += model.activation_sizes[11];
        model.activations.ln2_rstd = model.activations_memory[iter .. iter + model.activation_sizes[12]];
        iter += model.activation_sizes[12];
        model.activations.fch = model.activations_memory[iter .. iter + model.activation_sizes[13]];
        iter += model.activation_sizes[13];
        model.activations.fch_gelu = model.activations_memory[iter .. iter + model.activation_sizes[14]];
        iter += model.activation_sizes[14];
        model.activations.fcproj = model.activations_memory[iter .. iter + model.activation_sizes[15]];
        iter += model.activation_sizes[15];
        model.activations.residual3 = model.activations_memory[iter .. iter + model.activation_sizes[16]];
        iter += model.activation_sizes[16];
        model.activations.lnf = model.activations_memory[iter .. iter + model.activation_sizes[17]];
        iter += model.activation_sizes[17];
        model.activations.lnf_mean = model.activations_memory[iter .. iter + model.activation_sizes[18]];
        iter += model.activation_sizes[18];
        model.activations.lnf_rstd = model.activations_memory[iter .. iter + model.activation_sizes[19]];
        iter += model.activation_sizes[19];
        model.activations.logits = model.activations_memory[iter .. iter + model.activation_sizes[20]];
        iter += model.activation_sizes[20];
        model.activations.probs = model.activations_memory[iter .. iter + model.activation_sizes[21]];
        iter += model.activation_sizes[21];
        model.activations.losses = model.activations_memory[iter .. iter + model.activation_sizes[22]];
        iter += model.activation_sizes[22];

        model.inputs = try allocator.alloc(u32, B * T);
        model.targets = try allocator.alloc(u32, B * T);

        std.debug.print("num_activations: {}\n", .{model.num_activations});
    } else {
        if ((B > model.batch_size) or (T != model.seq_len)) {
            std.debug.print("Batch size or sequence length mismatch\n", .{});
            return;
        }
    }

    @memset(model.activations_memory, 0);

    if (model.inputs.len > inputs.len) {
        @memcpy(model.inputs, inputs);
    }

    if (targets.len != 0) {
        @memcpy(model.targets, targets);
    } else {
        model.inputs = try allocator.realloc(model.inputs, inputs.len);
        @memcpy(model.targets, targets);
    }
    // @TODO: attemping encoder forward pass
    // const vec_size: usize = 8;
    // if ((C % vec_size == 0) and (C > vec_size)) {
    //     encoder_forward_vec(vec_size, model.activations.encoded, inputs, model.params.word_token_embeddings, B, T, C);
    // }
    encoder_forward(model.activations.encoded, inputs, model.params.word_token_embeddings, model.params.word_position_embeddings, B, T, C);

    //////////
    // const params: ParameterTensors = model.params; // ???
    // const acts: ActivationTensors = model.activations; // ???
    var residual: []f32 = undefined;
    for (0..L) |l| {
        std.debug.print("in 0..L\n", .{});
        residual = if (1 == 0) model.activations.encoded else model.activations.residual3[(l - 1) * B * T * C ..];

        // weights for this layer
        const l_ln1w = model.params.layer_norm_weights_layer_1[l * C ..];
        const l_ln1b = model.params.layer_norm_biases_layer_1[l * C ..];
        // @TODO: commented cuz unimplemented and compiler whining
        // const l_qkvw = params.qkvw[l * 3 * C * C];
        // const l_qkvb = params.qkvb[l * 3 * C];
        // const l_attprojw = params.attention_projection_weights[l * C * C];
        // const l_attprojb = params.attention_projection_biases[l * C];

        // activations for this layer
        // TODO: continue here
        const l_ln1 = model.activations.ln1[l * B * T * C ..];
        const l_ln1_mean = model.activations.ln1_mean[l * B * T * C ..];
        const l_ln1_rstd = model.activations.ln1_rstd[l * B * T * C ..];
        // @TODO: commented cuz unused and compiler whining
        // const l_qkv = acts.qkv[l * B * T * 3 * C];
        // const l_atty = acts.attention_output[l * B * T * C];
        // const l_preatt = acts.pre_attention[l * B * NH * T * T];
        // const l_att = acts.attention[l * B * NH * T * T];
        // const l_attproj = acts.attention_projection[l * B * T * C];
        // const l_residual2 = acts.residual2[l * B * T * C];
        // const l_ln2 = acts.layer_norm2[l * B * T * C];
        // const l_ln2_mean = acts.layer_norm2_mean[l * B * T * C];
        // const l_ln2_rstd = acts.layer_norm2_rstd[l * B * T * C];
        // const l_fch = acts.fully_connected_hidden[l * B * T * 4 * C];
        // const l_fch_gelu = acts.fully_connected_hidden_gelu[l * B * T * 4 * C];
        // const l_fcproj = acts.fully_connected_projection[l * B * T * C];
        // const l_residual3 = acts.residual3[l * B * T * C];

        layer_norm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        // matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        // attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        // matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        // residual_forward(l_residual2, residual, l_attproj, B*T*C);
        // layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        // matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        // gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        // matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        // residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }

    residual = model.activations.residual3[(L - 1) * B * T * C ..]; // last residual is in residual3
    layer_norm_forward(model.activations.lnf, model.activations.lnf_mean, model.activations.lnf_rstd, residual, model.params.layer_norm_weights_layer_1, model.params.layer_norm_biases_layer_1, B, T, C);
    // matmul_forward(acts.logits, acts.lnf, params.wte, B, T, C, V);
    //

    // residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    // layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    // matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, V);
    // softmax_forward(acts.probs, acts.logits, B, T, V);
    // if (targets != NULL) {
    //    crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, V);//    float
    //    // C CODE BELOW
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

// fn mallocAndPointActivations(allocator: std.mem.Allocator, acts: *ActivationTensors, act_sizes: [NUM_ACTIVATION_TENSORS]usize, acts_memory: *[]f32) !void {
//     var total_size: usize = 0;
//     for (act_sizes) |size| {
//         total_size += size;
//     }
//
//     acts_memory.* = try allocator.alloc(f32, total_size);
//     var memory = acts_memory.*;
//
//     var offset: usize = 0;
//     acts.encoded = memory[offset .. offset + act_sizes[0]];
//     offset += act_sizes[0];
//     acts.ln1 = memory[offset .. offset + act_sizes[1]];
//     offset += act_sizes[1];
//     acts.ln1_mean = memory[offset .. offset + act_sizes[2]];
//     offset += act_sizes[2];
//     acts.ln1_rstd = memory[offset .. offset + act_sizes[3]];
//     offset += act_sizes[3];
//     acts.qkv = memory[offset .. offset + act_sizes[4]];
//     offset += act_sizes[4];
//     acts.atty = memory[offset .. offset + act_sizes[5]];
//     offset += act_sizes[5];
//     acts.preatt = memory[offset .. offset + act_sizes[6]];
//     offset += act_sizes[6];
//     acts.att = memory[offset .. offset + act_sizes[7]];
//     offset += act_sizes[7];
//     acts.attproj = memory[offset .. offset + act_sizes[8]];
//     offset += act_sizes[8];
//     acts.residual2 = memory[offset .. offset + act_sizes[9]];
//     offset += act_sizes[9];
//     acts.ln2 = memory[offset .. offset + act_sizes[10]];
//     offset += act_sizes[10];
//     acts.ln2_mean = memory[offset .. offset + act_sizes[11]];
//     offset += act_sizes[11];
//     acts.ln2_rstd = memory[offset .. offset + act_sizes[12]];
//     offset += act_sizes[12];
//     acts.fch = memory[offset .. offset + act_sizes[13]];
//     offset += act_sizes[13];
//     acts.fch_gelu = memory[offset .. offset + act_sizes[14]];
//     offset += act_sizes[14];
//     acts.fcproj = memory[offset .. offset + act_sizes[15]];
//     offset += act_sizes[15];
//     acts.residual3 = memory[offset .. offset + act_sizes[16]];
//     offset += act_sizes[16];
//     acts.lnf = memory[offset .. offset + act_sizes[17]];
//     offset += act_sizes[17];
//     acts.lnf_mean = memory[offset .. offset + act_sizes[18]];
//     offset += act_sizes[18];
//     acts.lnf_rstd = memory[offset .. offset + act_sizes[19]];
//     offset += act_sizes[19];
//     acts.logits = memory[offset .. offset + act_sizes[20]];
//     offset += act_sizes[20];
//     acts.probs = memory[offset .. offset + act_sizes[21]];
//     offset += act_sizes[21];
//     acts.losses = memory[offset .. offset + act_sizes[22]];
// }

// @TODO: vectorized version fn encoder_forward_vec(comptime...)
fn layer_norm_forward(out: []f32, mean: []f32, rstd: []f32, inp: []f32, weight: []f32, bias: []f32, B: u32, T: u32, C: u32) void {
    const eps = 1e-5;
    for (0..B) |b| {
        for (0..T) |t| {
            const x = inp[b * T * C + t * C ..];
            var m: f32 = 0.0;
            for (0..C) |i| {
                m += x[i];
            }
            m /= @floatFromInt(C);
            var v: f32 = 0.0;
            for (0..C) |i| {
                const xshift = x[i] - m;
                v += xshift * xshift;
            }
            v /= @floatFromInt(C);
            const s = 1.0 / std.math.sqrt(v + eps);
            var out_bt = out[b * T * C + t * C ..];
            for (0..C) |i| {
                const n = (s * (x[i] - m));
                const o = n * weight[i] + bias[i];
                out_bt[i] = o;
            }
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

// @TODO: rawdogged straight from Karpathy -- zig tricks?
fn encoder_forward(out: []f32, inp: []u32, wte: []f32, wpe: []f32, B: u32, T: u32, C: u32) void {
    std.debug.print("encoder_forward\n", .{});
    for (0..B) |b| {
        for (0..T) |t| {
            // seek to the output position in out[b,t,:]
            var out_bt = out[b * T * C + t * C ..];
            // get the index of the token at inp[b, t]
            const ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            const wte_ix = wte[ix * C ..];
            // seek to the position in wpe corresponding to the position
            const wpe_t = wpe[t * C ..];
            // add the two vectors and store the result in out[b,t,:]
            for (0..C) |i| {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

// fn matmul_forward(logits: []f32, lnf: []f32, wte: []f32, B: u32, T: u32, C: u32, V: u32) void {}

// TODO: old mess to clean up
// pub fn gelu(x: f32) f32 {
//     return 0.5 * x * (1.0 + std.math.tan(std.math.sqrt(2.0 / std.math.pi) * (x + 0.044715 * pow(x, 3.9))));
// }

// TODO: old mess to clean up
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

    model.config = config;

    // TODO: breakdown and explanation for newbies
    // at this point I might as well turn the repo into a full-on breakdown of a transformer's forward pass
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
    std.debug.print("file filehandle from read_n: {}", .{file});

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

    loader.tokens_file = std.fs.cwd().openFile(filepath, .{ .mode = .read_only }) catch {
        return FileOpenError.FileTooSmall;
    };

    const file_size = try loader.tokens_file.getEndPos();
    std.debug.print("{} \n", .{file_size});
    if (file_size == 0) {
        return error.FileEmpty;
    }
    loader.file_size = file_size;
    loader.current_position = 0;
    loader.batch = try allocator.alloc(u32, (B * T + 1));
    loader.inputs = loader.batch;
    loader.targets = loader.batch[1..];
    loader.num_batches = file_size / (B * T * @sizeOf(u32));
}

fn data_loader_reset(loader: *DataLoader) void {
    loader.current_position = 0;
}

fn data_loader_next_batch(loader: *DataLoader) !void {
    std.debug.print("File handle: {any}\n", .{loader.tokens_file});
    std.debug.print("File size: {} bytes\n", .{loader.file_size});
    std.debug.print("Next batch size: {} bytes\n", .{((loader.B * loader.T + 1) * @sizeOf(u32))});
    std.debug.print("Batch, token len: {}, {}\n", .{ loader.B, loader.T });

    const B: u32 = loader.B;
    const T: u32 = loader.T;

    if ((loader.current_position + (B * T + 1) * @sizeOf(u32)) > loader.file_size) {
        loader.current_position = 0;
    }

    std.debug.print("Current position: {}\n", .{loader.current_position});

    try loader.tokens_file.seekTo(loader.current_position);
    const t = B * T + 1;

    for (0..t) |i| {
        std.debug.print("reading item {}/{}\n", .{ i + 1, t });
        loader.batch[i] = try loader.tokens_file.reader().readInt(u32, builtin.cpu.arch.endian());
    }

    loader.current_position += (B * T + 1) * @sizeOf(u32);
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
