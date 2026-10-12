// C[m,n] = sum_k A[m,k] * B[n,k]  (torch Linear without bias). K=768, N=2 -> SPIRVSubgroupReduce.
func.func @main(%a: tensor<2x768xf32>, %b: tensor<2x768xf32>) -> tensor<2x2xf32> {
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<2x2xf32>
  %init = linalg.fill ins(%zero : f32) outs(%e : tensor<2x2xf32>) -> tensor<2x2xf32>
  %r = linalg.generic {
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<2x768xf32>, tensor<2x768xf32>) outs(%init : tensor<2x2xf32>) {
    ^bb0(%x: f32, %y: f32, %acc: f32):
      %p = arith.mulf %x, %y : f32
      %s = arith.addf %acc, %p : f32
      linalg.yield %s : f32
  } -> tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}
