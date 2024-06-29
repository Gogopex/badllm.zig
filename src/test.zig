const std = @import("std");
const mem = std.mem;
const builtin = @import("builtin");
const posix = std.posix;

const NUM_ACTIVATION_TENSORS = 23;
const NUM_PARAMETER_TENSORS = 16;

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

// TODO: list all all acronyms here instead
// link to multiheaded attention, wte, wpe, layer norm, attention core concepts
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

    const C = model.config.n_channels;
    const V = model.config.vocab_size;
    const max_T = model.config.max_seq_len;
    const L = model.config.n_layer;

    // @TODO: load debug file
    const state_file = try std.fs.cwd().openFile("gpt2_124M_debug_state.bin", .{});
    defer state_file.close();

    var state_header: [256]i32 = undefined;
    _ = try state_file.readAll(mem.asBytes(&state_header));

    if (state_header[0] != 20240327) {
        return error.BadMagicStateFile;
    }

    if (state_header[1] != 1) {
        return error.BadVersionStateFile;
    }

    const B = state_header[2]; // batch size, e.g. 4
    const T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)

    std.debug.print("[State]\n", .{});
    std.debug.print("batch_size: {}\n", .{B});
    std.debug.print("seq_len: {}\n", .{T});

    var expected_grads: ParameterTensors = undefined;
    const expected_grads_memory = malloc_and_point_parameters(allocator, &expected_grads, model.param_sizes);
    defer allocator.free(expected_grads_memory);

    // Assuming mallocAndPointParameters is implemented elsewhere
    try malloc_and_point_parameters(allocator, &expected_grads, model.param_sizes, expected_grads_memory);

    // Inputs and expected outputs, only used for error checking
    const x = try allocator.alloc(i32, B * T);
    defer allocator.free(x);

    const y = try allocator.alloc(i32, B * T);
    defer allocator.free(y);

    const expected_logits = try allocator.alloc(f32, B * T * V);
    defer allocator.free(expected_logits);

    const expected_loss = try allocator.alloc(f32, 1);
    defer allocator.free(expected_loss);

    // Read reference information
    _ = try state_file.readAll(mem.sliceAsBytes(x));
    _ = try state_file.readAll(mem.sliceAsBytes(y));
    _ = try state_file.readAll(mem.sliceAsBytes(expected_logits));
    _ = try state_file.readAll(mem.sliceAsBytes(expected_loss));
    _ = try state_file.readAll(mem.sliceAsBytes(expected_grads_memory));

    // @TODO: load expected grads
    // const expected_grads_memory = malloc_and_point_parameters(allocator, &expected_grads, model.param_sizes);

    var all_ok = true;
    const losses = [10]f32;

    for (0..10) |step| {
        gpt2_forward(allocator, &model, model.inputs, model.targets, 1, 3) catch |err| {
            std.debug.print("Error in forward pass: {}\n", .{err});
        };
        gpt2_zero_grad(&model);
        gpt2_backward(allocator, &model) catch |err| {
            std.debug.print("Error in backward pass: {}\n", .{err});
        };

        if (step == 0) {
            const logits_ok = 1;
            for (0..(B * T * V)) |i| {
                if (i < 3) {
                    // @TODO: expected_logits[i
                    std.debug.print("logits[{}]: {}\n", .{ i, model.activations.logits[i] });
                }
                if (@abs(model.activations.logits[i] - expected_logits[i]) > 1e-2) {
                    std.debug.print("logits[{}] mismatch: expected: {}, got: {}\n", .{ i, expected_logits[i], model.activations.logits[i] });
                    logits_ok = false;
                    break;
                }
            }
            all_ok = all_ok and logits_ok;

            if (@abs(model.mean_loss - expected_loss) >= 1e-2) {
                std.debug.print("mean loss mismatch: expected: {}, got: {}\n", .{ expected_loss, model.mean_loss });
                all_ok = false;
            } else {
                std.debug.print("loss is OK: {}, {}\n", .{ model.mean_loss, expected_loss });
            }

            // @TODO check gradients like llm.c
            const grads_ok: [16]i32 = undefined;
            const grads = model.gradients;
            grads_ok[0] = checkTensor(grads.word_token_embeddings, expected_grads.word_token_embeddings, V * C, "word_token_embeddings");
            grads_ok[1] = checkTensor(grads.word_position_embeddings, expected_grads.word_position_embeddings, max_T * C, "word_position_embeddings");
            grads_ok[2] = checkTensor(grads.layer_norm_weights_layer_1, expected_grads.layer_norm_weights_layer_1, L * C, "layer_norm_weights_layer_1");
            grads_ok[3] = checkTensor(grads.layer_norm_biases_layer_1, expected_grads.layer_norm_biases_layer_1, L * C, "layer_norm_biases_layer_1");
            grads_ok[4] = checkTensor(grads.qkvw, expected_grads.qkvw, L * 3 * C * C, "qkvw");
            grads_ok[5] = checkTensor(grads.qkvb, expected_grads.qkvb, L * 3 * C, "qkvb");
            grads_ok[6] = checkTensor(grads.attention_projection_weights, expected_grads.attention_projection_weights, L * C * C, "attention_projection_weights");
            grads_ok[7] = checkTensor(grads.attention_projection_biases, expected_grads.attention_projection_biases, L * C, "attention_projection_biases");
            grads_ok[8] = checkTensor(grads.layer_norm_weights_layer_2, expected_grads.layer_norm_weights_layer_2, L * C, "layer_norm_weights_layer_2");
            grads_ok[9] = checkTensor(grads.layer_norm_biases_layer_2, expected_grads.layer_norm_biases_layer_2, L * C, "layer_norm_biases_layer_2");
            grads_ok[10] = checkTensor(grads.feed_forward_weights, expected_grads.feed_forward_weights, L * 4 * C * C, "feed_forward_weights");
            grads_ok[11] = checkTensor(grads.feed_forward_biases, expected_grads.feed_forward_biases, L * 4 * C, "feed_forward_biases");
            grads_ok[12] = checkTensor(grads.feed_forward_projection_weights, expected_grads.feed_forward_projection_weights, L * C * 4 * C, "feed_forward_projection_weights");
            grads_ok[13] = checkTensor(grads.feed_forward_project_biases, expected_grads.feed, L * C, "feed_forward_project_biases");
            grads_ok[14] = checkTensor(grads.final_layer_norm_weights, expected_grads.final_layer_norm_weights, C, "final_layer_norm_weights");
            grads_ok[15] = checkTensor(grads.final_layer_norm_biases, expected_grads.final_layer_norm_biases, "final_layer_norm_biases");
            for (0..16) |i| {
                all_ok = all_ok and grads_ok[i];
            }
        }
        gpt2_update(&model, 1e-4, 0.9, 0.999, 1e-8, 0.01, step + 1) catch |err| {
            std.debug.print("Error in update: {}\n", .{err});
        };
        std.debug.print("step {}: loss {}", .{ step, model.mean_loss });
        losses[step] = model.mean_loss;
    }

    // compare losses
    const expected_losses = [_]f32{
        5.270007133483887,
        4.059706687927246,
        3.3751230239868164,
        2.8007826805114746,
        2.315382242202759,
        1.8490285873413086,
        1.3946564197540283,
        0.9991465210914612,
        0.6240804195404053,
        0.37651097774505615,
    };

    for (0..10) |i| {
        if (@abs(losses[i] - expected_losses[i]) >= 1e-2) {
            std.debug.print("loss[{}] mismatch: expected: {}, got: {}\n", .{ i, expected_losses[i], losses[i] });
            all_ok = false;
        } else {
            std.debug.print("loss[{}] is OK: {}, {}\n", .{ i, losses[i], expected_losses[i] });
        }
    }

    std.debug.print("overall ok: {}", .{all_ok});
}

fn gpt2_update(model: *GPT2, learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32, step: u32) !void {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    if (model.m.len == 0) {
        model.m = try model.allocator.alloc(f32, model.num_parameters);
        model.v = try model.allocator.alloc(f32, model.num_parameters);
    }

    for (0..model.num_parameters) |i| {
        const param = model.params_memory[i];
        const grad = model.gradients_memory[i];
        const m = beta1 * model.m[i] + (1.0 - beta1) * grad;
        const v = beta2 * model.v[i] + (1.0 - beta2) * grad * grad;
        const m_hat = m / (1.0 - std.math.pow(f32, beta1, step));
        const v_hat = v / (1.0 - std.math.pow(f32, beta2, step));

        model.m_memory[i] = m;
        model.v_memory[i] = v;
        model.params_memory[i] = param - learning_rate * (m_hat / (std.math.sqrt(v_hat) + epsilon) + weight_decay * param);
    }
}

pub fn checkTensor(a: []const f32, b: []const f32, n: i32, label: []const u8) bool {
    const print_upto = 5;
    var ok = true;

    std.debug.print("{s}\n", .{label});

    for (0..n) |i| {
        if (@abs(a[i] - b[i]) <= 1e-2) {
            if (i < print_upto) {
                std.debug.print("OK", .{});
            }
        } else {
            if (i < print_upto) {
                std.debug.print("NOT OK", .{});
            }
            ok = false;
        }
        if (i < print_upto) {
            std.debug.print("a[{}]: {}, b[{}]: {}\n", .{ i, a[i], i, b[i] });
        }
    }

    // Print the final result
    if (ok) {
        std.debug.print("TENSOR OK\n", .{});
    } else {
        std.debug.print("TENSOR NOT OK\n", .{});
    }

    return ok;
}

fn load_debug_state(allocator: std.mem.Allocator, model: *GPT2, V: u32) !std.fs.File {
    const state_file = try std.fs.cwd().openFile("gpt2_124M_debug_state.bin", .{});

    var state_header: [256]i32 = undefined;
    _ = try state_file.readAll(mem.asBytes(&state_header));

    if (state_header[0] != 20240327) {
        return error.BadMagicStateFile;
    }

    if (state_header[1] != 1) {
        return error.BadVersionStateFile;
    }

    const B = state_header[2]; // batch size, e.g. 4
    const T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
    //
    // fn gpt2_update(

    std.debug.print("[State]\n", .{});
    std.debug.print("batch_size: {}\n", .{B});
    std.debug.print("seq_len: {}\n", .{T});

    var expected_grads: ParameterTensors = undefined;
    const expected_grads_memory = try allocator.alloc(f32, model.num_parameters);
    defer allocator.free(expected_grads_memory);

    // Assuming mallocAndPointParameters is implemented elsewhere
    try malloc_and_point_parameters(allocator, &expected_grads, model.param_sizes, expected_grads_memory);

    // Inputs and expected outputs, only used for error checking
    const x = try allocator.alloc(i32, B * T);
    defer allocator.free(x);

    const y = try allocator.alloc(i32, B * T);
    defer allocator.free(y);

    const expected_logits = try allocator.alloc(f32, B * T * V);
    defer allocator.free(expected_logits);

    const expected_loss = try allocator.alloc(f32, 1);
    defer allocator.free(expected_loss);

    // Read reference information
    _ = try state_file.readAll(mem.sliceAsBytes(x));
    _ = try state_file.readAll(mem.sliceAsBytes(y));
    _ = try state_file.readAll(mem.sliceAsBytes(expected_logits));
    _ = try state_file.readAll(mem.sliceAsBytes(expected_loss));
    _ = try state_file.readAll(mem.sliceAsBytes(expected_grads_memory));

    return state_file;
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
        @memset(model.gradients_activations_memory, 0);
    }
}

fn gpt2_backward(allocator: std.mem.Allocator, model: *GPT2) !void {
    if (model.mean_loss == -1.0) {
        std.debug.print("Error: must forward with targets before applying backward\n", .{});
    }

    if (model.gradients_memory.len == 0) {
        // model.gradients_memory = try allocator.alloc(f32, model.gradients);
        // model.gradients_activations_memory = try allocator.alloc(f32, model.activation_sizes);
        model.gradients_memory = malloc_and_point_parameters(allocator, &model.gradients, model.param_sizes);
        model.gradients_activations_memory = malloc_and_point_activations(allocator, &model.gradients_activations, model.activation_sizes);
    }

    const B = model.batch_size;
    const T = model.seq_len;
    const V = model.config.vocab_size;
    const L = model.config.n_layer;
    const C = model.config.n_channels;
    const NH = model.config.n_head;

    const params = model.params;
    const grads = model.gradients;
    const acts = model.activations;
    const grads_acts = model.gradients_activations;

    const dloss_mean = 1.0 / (B * T);
    for (0..B * T) |i| {
        grads_acts.losses[i] = dloss_mean;
    }

    var dbiases: []f32 = undefined;
    dbiases.len = 0;

    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model.targets, B, T, V);
    matmul_backward(grads_acts.lnf, grads.wte, dbiases, grads_acts.logits, acts.lnf, params.wte, B, T, C, V);
    const residual = acts.residual3 + (L - 1) * B * T * C;
    const dresidual = grads_acts.residual3 + (L - 1) * B * T * C;
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

    for (0..L - 1) |l| {
        // @TODO: PICK UP BACK FROM HERE
    }
}

pub fn layernorm_backward(dinp: []f32, dweight: []f32, dbias: []f32, dout: []f32, inp: []f32, inp: []f32, weight: []f32, mean: []f32, rtsd: []f32, B: u32, T: u32, C: u32) void {
    for (0..B) |b| {
        for (0..T) |t| {
            const dout_bt = dout[b * T * C + t * C];
            const inp_bt = dinp[b * T * C + t * C];
            const dinp_bt = dinp[b * T * C + t * C];
            const mean_bt = mean[b * T + t];
            const rstd_bt = rtsd[b * T + t];

            var dnorm_mean = 0.0;
            var dnorm_norm_mean = 0.0;
            for (0..C) |i| {
                const norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                const dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean /= C;
            dnorm_norm_mean /= C;

            for (0..C) |i| {
                const norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                const dnorm_i = weight[i] * dout_bt[i];
                const dnorm = dnorm_i * rstd_bt;
                const dinp_i = dnorm - dnorm_mean - norm_bti * dnorm_rstd;
                dinp_bt[i] += dinp_i;
                dweight[i] += dinp_i * norm_bti;
                dbias[i] += dnorm_i;
                var dval = 0.0;
                dval += dnorm_i;
                dval -= dnorm_mean;
                dval -= norm_bti * dnorm_norm_mean;
                dval *= rstd_bt;
                dinp_bt[i] += dval;
            }
        }
    }
}

pub fn crossentropy_softmax_backward(
    dlogits: []f32,
    dlosses: []const f32,
    probs: []const f32,
    targets: []const i32,
    B: usize,
    T: usize,
    V: usize,
) void {
    const BT = B * T;

    for (0..BT) |bt| {
        const base_index = bt * V;
        const dloss = dlosses[bt];
        const target: usize = @intCast(targets[bt]);

        for (0..V) |i| {
            const index = base_index + i;
            const p = probs[index];
            const indicator: f32 = if (i == target) 1.0 else 0.0;
            dlogits[index] += (p - indicator) * dloss;
        }
    }
}

pub fn matmul_backward(
    dinp: []f32,
    dweight: []f32,
    dbias: ?[]f32,
    dout: []const f32,
    inp: []const f32,
    weight: []const f32,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) !void {
    for (0..B) |b| {
        for (0..T) |t| {
            const dout_bt = dout[b * T * OC + t * OC];
            const dinp_bt = dinp[b * T * C + t * C];
            for (0..OC) |o| {
                const wrow = weight[o * C];
                const d = dout_bt[o];
                for (0..C) |c| {
                    dinp_bt[c] += d * wrow[c];
                }
            }
        }
    }

    for (0..OC) |o| {
        for (0..B) |b| {
            for (0..T) |t| {
                const dout_bt = dout[b * T * OC + t * OC];
                const inp_bt = inp[b * T * C + t * C];
                const dwrow = dweight[o * C];
                const d = dout_bt[o];
                if (dbias != null) {
                    dbias[o] += d;
                }
                for (0..C) |c| {
                    dwrow[c] += d * inp_bt[c];
                }
            }
        }
    }
}

fn malloc_and_point_activations(allocator: mem.Allocator, acts: *ActivationTensors, act_sizes: []const usize) ![]f32 {
    var num_activations: usize = 0;
    for (act_sizes[0..NUM_ACTIVATION_TENSORS]) |size| {
        num_activations += size;
    }
    // Allocate all activations at once
    const acts_memory = try allocator.alloc(f32, num_activations);
    // Define an array of pointers to the tensor fields in ActivationTensors
    const ptrs = [_]*?[*]f32{
        &acts.encoded,   &acts.ln1,       &acts.ln1_mean, &acts.ln1_rstd,
        &acts.qkv,       &acts.atty,      &acts.preatt,   &acts.att,
        &acts.attproj,   &acts.residual2, &acts.ln2,      &acts.ln2_mean,
        &acts.ln2_rstd,  &acts.fch,       &acts.fch_gelu, &acts.fcproj,
        &acts.residual3, &acts.lnf,       &acts.lnf_mean, &acts.lnf_rstd,
        &acts.logits,    &acts.probs,     &acts.losses,
    };
    var acts_memory_iterator: [*]f32 = acts_memory.ptr;
    for (ptrs, act_sizes[0..NUM_ACTIVATION_TENSORS]) |ptr, size| {
        ptr.* = acts_memory_iterator;
        acts_memory_iterator += size;
    }
    return acts_memory;
}

fn malloc_and_point_parameters(allocator: mem.Allocator, params: *ParameterTensors, param_sizes: []const usize) ![]f32 {
    var num_parameters: usize = 0;
    for (param_sizes[0..NUM_PARAMETER_TENSORS]) |size| {
        num_parameters += size;
    }
    // Allocate all parameters at once
    const params_memory = try allocator.alloc(f32, num_parameters);
    // Define an array of pointers to the tensor fields in ParameterTensors
    const ptrs = [_]*?[*]f32{
        &params.wte,     &params.wpe,     &params.ln1w,     &params.ln1b,
        &params.qkvw,    &params.qkvb,    &params.attprojw, &params.attprojb,
        &params.ln2w,    &params.ln2b,    &params.fcw,      &params.fcb,
        &params.fcprojw, &params.fcprojb, &params.lnfw,     &params.lnfb,
    };
    var params_memory_iterator: [*]f32 = params_memory.ptr;
    for (ptrs, param_sizes[0..NUM_PARAMETER_TENSORS]) |ptr, size| {
        ptr.* = params_memory_iterator;
        params_memory_iterator += size;
    }
    return params_memory;
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

    std.debug.print("activation memory length: {}\n", .{model.activations_memory.len});
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
        std.debug.print("B, model.batch_size: {}, {}\n", .{ B, model.batch_size });
        std.debug.print("T, model.seq_len: {}, {}\n", .{ T, model.seq_len });
        std.debug.assert(B > model.batch_size);
        std.debug.assert(T != model.seq_len);
        std.debug.assert((B > model.batch_size) or (T != model.seq_len));
        if ((B > model.batch_size) or (T != model.seq_len)) {
            std.debug.print("Batch size or sequence length mismatch\n", .{});
            return;
        }
    }

    std.debug.print("model.inputs.len: {}, inputs len: {}", .{ model.inputs.len, inputs.len });

    if (model.inputs.len > inputs.len) {
        @memcpy(model.inputs, inputs);
    }

    std.debug.print("after memcpy", .{});

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
    std.debug.assert(model.config.n_channels == C);
    encoder_forward(model.activations.encoded, inputs, model.params.word_token_embeddings, model.params.word_position_embeddings, B, T, C);

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
            var out_bt = out[b * T * C + t * C ..];
            // Get the index of the token at inp[b, t]
            const ix: u32 = inp[b * T + t];
            // Seek to the position in wte corresponding to the token
            const wte_ix = wte[ix * C ..];
            // Seek to the position in wpe corresponding to the position
            std.debug.print("wpe len: {}\n", .{wpe.len});
            std.debug.print("t, C: {}, {}\n", .{ t, C });
            const wpe_t = wpe[t * C ..];
            std.debug.print("wpe_t.len: {}\n", .{wpe_t.len});

            for (0..C) |c| {
                out_bt[c] = wte_ix[c] + wpe_t[c];
            }
        }
    }
}

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

    std.debug.print("config n_channels: {}\n", .{config.n_channels});
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

    // @TOOD: llm.c artifacts
    // // read in all the parameters from file
    // model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    // fread(model->params_memory, sizeof(float), num_parameters, model_file);
    // fclose(model_file);
    //
    // // other inits
    model.activations_memory = std.mem.zeroes([]f32);

    // @memset(model.activations_memory, 0);
    model.gradients_memory = undefined;
    model.m_memory = undefined;
    model.v_memory = undefined;
    model.gradients_activations_memory = undefined;
    model.inputs = undefined;
    model.targets = undefined;
    model.batch_size = 0;
    model.seq_len = 0;
    model.mean_loss = -1.0; // -1.0 will designate no loss
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
