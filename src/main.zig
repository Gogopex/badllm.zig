const std = @import("std");

const GPT2Config = struct {
    max_seq_len: u32, // Maximum sequence length eg 1024
    vocab_size: u32, // Vocabulary size eg 50257
    n_embd: u32, // Embedding dimension eg 768
    n_layer: u32, // Number of layers eg 12
    n_head: u32, // Number of attention heads eg 12
    n_channels: u32, // Number of channels in the MLP eg 768
    padded_vocab_size: usize,
};

const GPT2 = struct {
    config: GPT2Config,
    params: []f32,
    params_memory: []f32,
    activations: []f32,
    activations_memory: []f32,
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

    const default = GPT2{
        .config = GPT2Config{
            .max_seq_len = 0,
            .vocab_size = 0,
            .padded_vocab_size = 0,
            .n_layer = 0,
            .n_embd = 0,
            .n_head = 0,
            .n_channels = 0,
        },
        .params = &[_]f32{},
        .params_memory = &[_]f32{},
        .activations = &[_]f32{},
        .activations_memory = &[_]f32{},
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
// const TransformerBlock = struct {
//     word_token_embeddings: f32, // shape V, C where V is vocab size, C is embedding dims -- each word in the vocab is mapped to a vector of size C
//     word_position_embeddings: f32, // shape maxT, C -- maxT is maximum sequence length, C is embeddingdims -- adds positional info to the token embeddings
//     layer_norm_weights_layer_1: f32, // shape L, C -- L is the num of layers, C embedding dims
//     layer_norm_biases_layer_1: f32, // shape L, C -- L is the num of layers, C embedding dims
//     qkvw: f32, // shape L, 3C, C -- query key values weight projections for multiheaded attention -- L is num of layers, 3C is query/key/values concat, C is the embedding dims
//     qkvb: f32, // shape L, 3C -- query key values bias projections for multiheaded attention
//     attention_projection_weights: f32, // shape L, C, C -- weights of the concat output of the attention heads back to the embedding dimension
//     attention_projection_biases: f32, // shape L, C -- biases of the concat output of the attention heads back to the embedding dimension
//     layer_norm_weights_layer_2: f32, //
//     layer_norm_biases_layer_2: f32,
//     feed_forward_weights: f32, // shape L, 4C, C -- weights of the FFN
//     feed_forward_biases: f32, // shape L, 4C -- biases of the FFN
//     feed_forward_projection_weights: f32, // L, C, 4C -- weights for projecting the output of the FFN back to the embedding dimension
//     final_layer_norm_weights: f32, // shape C -- final weights for the final layer norm
//     final_layer_norm_biases: f32, // shape C -- final biases for the final layer norm
// };

pub fn main() !void {
    std.debug.print("Hello, world!\n", .{});

    const model = GPT2.default;

    if (build_gpt2_from_checkpoint(model, "gpt2_124M.bin")) |err| {
        std.debug.print("Error building GPT-2 model: {}\n", .{err});
        return;
    } else |_| {
        std.debug.print("GPT-2 model built successfully\n", .{});
    }

    // const pos_encodings = load_positional_encodings("gpt2_124M.bin", 1024, 768);

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

fn build_gpt2_from_checkpoint(model: GPT2, filepath: []const u8) []f32 {
    var file = std.fs.cwd().openFile(filepath, .{}) catch {
        std.debug.print("Error opening the model file\n", .{});
        std.posix.exit(1);
    };

    const header = std.mem.zeroes(i32);
    file.readAll(std.mem.bytesAsSlice(u8, header)) catch {
        std.debug.print("Error opening the model file\n", .{});
        std.posix.exit(1);
    };

    std.debug.print("Header: {}", .{header});

    if (header[0] != 20240326) {
        std.debug.print("Bad magic model file\n", .{});
        std.posix.exit(1);
    }

    if (header[1] != 3) {
        std.debug.print("Bad version in model file\n", .{});
        std.debug.print("---> HINT: try to re-run `python train_gpt2.py`\n", .{});
        std.posix.exit(1);
    }

    model.config.max_seq_len = @intCast(header[2]);
    model.config.vocab_size = @intCast(header[3]);
    model.config.n_layer = @intCast(header[4]);
    model.config.n_head = @intCast(header[5]);
    model.config.n_channels = @intCast(header[6]);
    model.config.padded_vocab_size = @intCast(header[7]);

    std.debug.print("[GPT-2]\n", .{});
    std.debug.print("max_seq_len: {}\n", .{model.config.max_seq_len});
    std.debug.print("vocab_size: {}\n", .{model.config.vocab_size});
    std.debug.print("padded_vocab_size: {}\n", .{model.config.padded_vocab_size});
    std.debug.print("num_layers: {}\n", .{model.config.num_layers});
    std.debug.print("num_heads: {}\n", .{model.config.num_heads});
    std.debug.print("channels: {}\n", .{model.config.channels});

    var num_parameters: usize = 0;
    const param_sizes = std.ArrayList(usize).init(std.heap.page_allocator);

    // TODO fill param_sizes with the sizes of the parameters

    fill_in_param_sizes(file, param_sizes);

    for (param_sizes.items) |size| {
        num_parameters += size;
    }

    std.debug.print("num_parameters: {}\n", .{num_parameters});

    model.params_memory = std.heap.page_allocator.create([]f32, num_parameters) catch {
        std.debug.print("Error allocating memory for the model parameters\n", .{});
        std.posix.exit(1);
    };

    file.close();

    model.activations_memory = null;
    model.gradients_memory = null;
    model.gradients_activations_memory = null;
    model.m_memory = null;
    model.v_memory = null;
    model.inputs = null;
    model.targets = null;
    model.batch_size = 0;
    model.seq_len = 0;
    model.mean_loss = -1.0;
}

fn fill_in_param_sizes(file: std.fs.File, param_sizes: std.ArrayList(usize)) void {
    const size = std.mem.zeroes(i32);
    file.readAll(std.mem.bytesAsSlice(u8, size)) catch {
        std.debug.print("Error reading the model file\n", .{});
        std.posix.exit(1);
    };
    param_sizes.append(@intCast(size));
}
// fn load_embeddings(filepath: []const u8, vocab_size: usize, embedding_dim: usize) []f32 {
//     return load_weights(filepath, vocab_size * embedding_dim);
// }
//
// fn load_positional_encodings(filepath: []const u8, seq_len: usize, embedding_dim: usize) []f32 {
//     return load_weights(filepath, seq_len * embedding_dim);
// }

// TODO design basic forward pass implementation
// load weights and embeddings
// multiheaded attention
// FFN
// add & norm

test "simple test" {
    try std.testing.expectEqual(1, 1);
}
