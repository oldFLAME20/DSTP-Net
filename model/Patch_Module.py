# import torch.nn as nn
# from .embedding import InputEmbedding
# from model.ours_memory_module import MemoryModule
#
# class PatchBranch(nn.Module):
#     def __init__(self, win_size, enc_in, d_model, n_heads, e_layers, d_ff, dropout, activation,
#                  patch_scales, n_memory_per_scale, shrink_thres, device, phase_type, dataset_name):
#         super(PatchBranch, self).__init__()
#         self.patch_scales = patch_scales
#         self.d_model = d_model
#         self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)
#         self.encoders = nn.ModuleList()
#         self.memories = nn.ModuleList()
#         for z in patch_scales:
#             effective_win_size = win_size // z
#             self.encoders.append(
#                 Encoder([EncoderLayer(AttentionLayer(effective_win_size, d_model, n_heads, dropout=dropout),
#                                       d_model, d_ff, dropout=dropout, activation=activation) for _ in range(e_layers)],
#                         norm_layer=nn.LayerNorm(d_model)))
#             mem_dataset_name_suffix = f"_patch_z{z}"
#             self.memories.append(
#                 MemoryModule(n_memory=n_memory_per_scale, fea_dim=d_model, shrink_thres=shrink_thres, device=device,
#                              phase_type=phase_type, dataset_name=f"{dataset_name}{mem_dataset_name_suffix}"))
#         fusion_input_dim = d_model * len(patch_scales)
#         self.fusion_net = nn.Sequential(nn.Linear(fusion_input_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
#         self.decoder = Decoder(d_model * 2, enc_in, d_ff, activation, dropout)
# ####这是原先可以运行的
#     def forward(self, x, memory_init_embeddings=None):
#         N, L, C_in = x.shape
#         queries_from_scales, attns_from_scales, mems_from_scales, recons_from_scales = {}, {}, {}, {}
#         for i, z in enumerate(self.patch_scales):
#             x_pooled = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=z, stride=z).permute(0, 2, 1) if z > 1 else x
#             pooled_embed = self.embedding(x_pooled)
#             patch_queries = self.encoders[i](pooled_embed)
#             if memory_init_embeddings and f"patch_z{z}" in memory_init_embeddings:
#                 self.memories[i].mem = nn.Parameter(memory_init_embeddings[f"patch_z{z}"])
#             mem_out = self.memories[i](patch_queries)
#             recons_queries_z = mem_out['output'].view(N, L // z, 2 * self.d_model)[:, :, self.d_model:]
#             queries_from_scales[z] = patch_queries
#             attns_from_scales[z] = mem_out['attn']
#             mems_from_scales[z] = self.memories[i].mem
#             # recons_from_scales[z] = F.interpolate(recons_queries_z.transpose(1, 2), size=L, mode='linear',
#             #                                       align_corners=False).transpose(1, 2)
#             # ==============================================================
#             # ===        使用我们自己的确定性 interpolate 函数         ===
#             # ==============================================================
#             # 旧代码:
#             # recons_from_scales[z] = F.interpolate(recons_queries_z.transpose(1, 2), size=L, mode='linear',
#             #                                       align_corners=False).transpose(1, 2)
#
#             # 新代码:
#             # 1. 确保输入是 [N, C, L] 格式
#             tensor_to_interpolate = recons_queries_z.transpose(1, 2)
#             # 2. 调用确定性函数
#             interpolated_tensor = deterministic_interpolate(tensor_to_interpolate, target_len=L)
#             # 3. 恢复形状
#             recons_from_scales[z] = interpolated_tensor.transpose(1, 2)
#             # ==============================================================
#
#         fusion_input_list = [recons_from_scales[z] for z in self.patch_scales]  # Fuse recons as per H-PAD figure
#         fusion_input = torch.cat(fusion_input_list, dim=-1)
#         final_queries = self.fusion_net(fusion_input)
#         final_mem_out = self.memories[0](final_queries)  # Use z=1 memory for final enhancement
#         final_recons_output = self.decoder(final_mem_out['output'])
#         return {'output': final_recons_output, 'queries': queries_from_scales, 'mems': mems_from_scales,
#                 'attns': attns_from_scales}
