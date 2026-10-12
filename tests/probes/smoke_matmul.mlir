func.func @main(%a: tensor<64x128xf32>, %b: tensor<128x32xf32>) -> tensor<64x32xf32> {
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<64x32xf32>
  %init = linalg.fill ins(%zero : f32) outs(%e : tensor<64x32xf32>) -> tensor<64x32xf32>
  %r = linalg.matmul ins(%a, %b : tensor<64x128xf32>, tensor<128x32xf32>) outs(%init : tensor<64x32xf32>) -> tensor<64x32xf32>
  return %r : tensor<64x32xf32>
}
