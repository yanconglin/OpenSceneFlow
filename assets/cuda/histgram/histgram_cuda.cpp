#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <vector>

void histgram_func(
    const at::Tensor &pts_src, 
    const at::Tensor &pts_dst, 
    const at::Tensor &cls_src, 
    at::Tensor &histgram_translation, 
    const float min_x, 
    const float min_y, 
    const float min_z, 
    const float max_x, 
    const float max_y, 
    const float max_z, 
    const int window_x, 
    const int window_y,
    const int window_z
    );
    
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("histgram_func", &histgram_func, "histgram (CUDA)");
}