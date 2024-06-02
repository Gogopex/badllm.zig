const std = @import("std");

// const Tensor = struct {
//     data: []f32,
//     rows: usize,
//     cols: usize,
//
//     pub fn init(_: usize, _: usize) Tensor {
//         unreachable;
//     }
// };
//
// const Layer = struct {
//     weights: Tensor,
//     biases: Tensor,
// };
//
// const MLP = struct {
//     layers: []Layer,
//
//     pub fn forward(_: *MLP, _: []f64) []f64 {
//         unreachable;
//     }
// };
//
// fn relu(x: f64) f64 {
//     return if (x > 0.0) x else 0.0;
// }
//
// fn sigmoid(x: f64) f64 {
//     return 1.0 / (1.0 + std.math.exp(-x));
// }
// fn tanh(x: f64) f64 {
//     return std.math.tanh(x);
// }
