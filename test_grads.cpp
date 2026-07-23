#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include "tensor.hpp"
#include "ops.hpp"
#include "optimizer.hpp"
#include "nn.hpp"
#include "vit.hpp"

using namespace vit;

void print_tensor(const vit::Tensor& t, const std::string& name, bool to_host = false) {
#ifdef USE_CUDA
    if (to_host) const_cast<vit::Tensor&>(t)->to_host();
#endif
    std::cout << name << ": ";
    for(int j = 0; j < std::min(10, t->cols); ++j) {
        std::cout << t->at(0, j) << " ";
    }
    std::cout << "\n";
}

int main() {
    std::mt19937 rng(42);
    ViTConfig cfg;
    cfg.image_size = 28; cfg.patch_size = 14;
    cfg.embed_dim = 16; cfg.depth = 1;
    cfg.num_heads = 2; cfg.num_classes = 10;
    
    VisionTransformer model(cfg, rng);
    std::vector<float> img(28 * 28, 0.5f);
    
    // Extracted from VisionTransformer::forward
    vit::Tensor patches = vit::patchify(img, cfg);
    vit::Tensor x = model.patch_embed.forward(patches);
    print_tensor(x, "Patch Embed", true);
    
    vit::Tensor cls = model.cls_token;
    x = vit::VisionTransformer::concat_rows_helper(cls, x);
    x = vit::add(x, model.pos_embed);
    print_tensor(x, "Pos Embed Added", true);
    
    // First block
    auto blk = model.blocks[0];
    vit::Tensor nx = blk->ln1.forward(x);
    print_tensor(nx, "LN1 Out", true);
    
    vit::Tensor Q = blk->attn.q_proj.forward(nx);
    vit::Tensor K = blk->attn.k_proj.forward(nx);
    vit::Tensor V = blk->attn.v_proj.forward(nx);
    print_tensor(Q, "Q Proj", true);
    print_tensor(K, "K Proj", true);
    
    vit::Tensor Qh = vit::slice_cols(Q, 0, 8);
    vit::Tensor Kh = vit::slice_cols(K, 0, 8);
    vit::Tensor scores = vit::scale(vit::matmul(Qh, vit::transpose(Kh)), 1.0f / std::sqrt(8.0f));
    print_tensor(scores, "Attention Scores", true);
    
    vit::Tensor attn = vit::softmax_rows(scores);
    print_tensor(attn, "Attention Softmax", true);
    
    vit::Tensor ax = blk->attn.forward(nx);
    print_tensor(ax, "Attention Out", true);
    
    x = vit::add(x, ax);
    x = vit::add(x, blk->mlp.forward(blk->ln2.forward(x)));
    print_tensor(x, "Block Out", true);
    
    vit::Tensor cls_out = vit::select_row(x, 0);
    vit::Tensor logits = model.final_ln.forward(cls_out);
    logits = model.head.forward(logits);
    print_tensor(logits, "Logits", true);
    
    return 0;
}
