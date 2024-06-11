const std = @import("std");

const FileOpenError = error{
    InvalidHeader,
    UnsupportedVersion,
    FileEmpty,
    FileTooSmall,
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

    var model = GPT2.default;
    build_model_from_file("gpt2_124M.bin", &model) catch |err| {
        std.debug.print("Error building model: {}\n", .{err});
        return;
    };

    // var tokenizer: Tokenizer = undefined;
    // if (build_tokenizer_from_vocab("gpt2_tokenizer.bin", &tokenizer)) |err| {
    //     std.debug.print("Error building tokenizer: {}\n", .{err});
    //     return;
    // } else |_| {
    //     std.debug.print("Tokenizer built successfully\n", .{});
    // }

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

const Tokenizer = struct { vocabulary_size: u32, init: bool, data: []u8 };

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

    std.debug.print("max_seq_len: {}, vocab_size: {}, n_layer: {}, n_head: {}, n_channels: {}", .{ config.max_seq_len, config.vocab_size, config.n_layer, config.n_head, config.n_channels });

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
fn read_n_from_checkpoint_file(filepath: []const u8, N: usize, comptime T: type, offset: usize) !std.ArrayList(f32) {
    var file = try std.fs.cwd().openFile(filepath, .{ .mode = .read_only });
    defer file.close();

    const file_size = try file.getEndPos();
    if (file_size < N * @sizeOf(T)) {
        return error.FileTooSmall;
    }
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

// TODO: design basic forward pass implementation
// load weights and embeddings
// multiheaded attention
// FFN
// add & norm

test "simple test" {
    try std.testing.expectEqual(1, 1);
}
