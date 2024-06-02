const std = @import("std");

// TODO: load data
// split into train and validation

// const GPT2Config = struct {
//     max_seq_len: u32, // Maximum sequence length eg 1024
//     vocab_size: u32, // Vocabulary size eg 50257
//     n_embd: u32, // Embedding dimension eg 768
//     n_layer: u32, // Number of layers eg 12
//     n_head: u32, // Number of attention heads eg 12
//     n_channels: u32, // Number of channels in the MLP eg 768
// };
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

fn load_weights(filepath: []const u8, num_weights: usize) []f32 {
    var file = std.fs.cwd().openFile(filepath, .{}) catch unreachable;
    // const filesize = file.size catch unreachable;
    // TODO assert?

    const buffer = std.heap.page_allocator.create([]f32, num_weights) catch unreachable;
    file.readAll(std.mem.bytesAsSlice(u8, buffer)) catch unreachable;

    return buffer;
}

fn load_embeddings(filepath: []const u8, vocab_size: usize, embedding_dim: usize) []f32 {
    return load_weights(filepath, vocab_size * embedding_dim);
}

fn load_positional_encodings(filepath: []const u8, seq_len: usize, embedding_dim: usize) []f32 {
    return load_weights(filepath, seq_len * embedding_dim);
}

// TODO design basic forward pass implementation
// load weights and embeddings
// multiheaded attention
// FFN
// add & norm

test "simple test" {
    try std.testing.expectEqual(1, 1);
}
