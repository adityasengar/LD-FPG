# models.py
# Description: PyTorch model definitions (HNO Encoder, Decoder2).

from common_imports import (
    torch, nn, F, ChebConv, # ChebConv from torch_geometric.nn
    logging, sys, Dict, List, Optional, Tuple, Any
)

# Custom project module imports (if any, math_utils is common)
from math_utils import build_decoder_mlp

logger = logging.getLogger(__name__)

class HNO(nn.Module):
    """HNO Encoder based on ChebConv layers."""
    def __init__(self, hidden_dim: int, K_cheb_order: int):
        super().__init__()
        self._debug_logged_forward = False
        self._debug_logged_repres = False
        logger.debug(f"Initializing HNO Encoder: hidden_dim={hidden_dim}, K={K_cheb_order}")
        sys.stdout.flush() # For early debug message visibility

        self.conv1 = ChebConv(in_channels=3, out_channels=hidden_dim, K=K_cheb_order)
        self.bano1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.conv2 = ChebConv(in_channels=hidden_dim, out_channels=hidden_dim, K=K_cheb_order)
        self.bano2 = nn.BatchNorm1d(num_features=hidden_dim)
        self.conv3 = ChebConv(in_channels=hidden_dim, out_channels=hidden_dim, K=K_cheb_order)
        self.bano3 = nn.BatchNorm1d(num_features=hidden_dim)
        self.conv4 = ChebConv(in_channels=hidden_dim, out_channels=hidden_dim, K=K_cheb_order)
        self.mlpRep = nn.Linear(hidden_dim, 3) # For direct coordinate reconstruction

    def forward_representation(self, x: torch.Tensor, edge_index: torch.Tensor, log_debug_shapes: bool = False) -> torch.Tensor:
        """Computes latent atom-wise embeddings Z."""
        if log_debug_shapes and not self._debug_logged_repres:
            logger.debug(f"[HNO.rep] Input x: {x.shape}, edge_index: {edge_index.shape}")
        x = x.float()
        x = F.leaky_relu(self.bano1(self.conv1(x, edge_index)))
        if log_debug_shapes and not self._debug_logged_repres: logger.debug(f"[HNO.rep] After conv1: {x.shape}")
        x = F.leaky_relu(self.bano2(self.conv2(x, edge_index)))
        if log_debug_shapes and not self._debug_logged_repres: logger.debug(f"[HNO.rep] After conv2: {x.shape}")
        x = F.relu(self.bano3(self.conv3(x, edge_index))) # ReLU for third
        if log_debug_shapes and not self._debug_logged_repres: logger.debug(f"[HNO.rep] After conv3: {x.shape}")
        x = self.conv4(x, edge_index)
        if log_debug_shapes and not self._debug_logged_repres: logger.debug(f"[HNO.rep] After conv4: {x.shape}")
        z_embeddings = F.normalize(x, p=2.0, dim=1) # L2 norm per atom
        if log_debug_shapes and not self._debug_logged_repres:
            logger.debug(f"[HNO.rep] Output Z (L2 norm): {z_embeddings.shape}")
            self._debug_logged_repres = True
        return z_embeddings

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, log_debug_shapes: bool = False) -> torch.Tensor:
        """Full pass including MLP for coordinate reconstruction."""
        z_embeddings = self.forward_representation(x, edge_index, log_debug_shapes=log_debug_shapes)
        reconstructed_coords = self.mlpRep(z_embeddings)
        if log_debug_shapes and not self._debug_logged_forward:
            logger.debug(f"[HNO.fwd] Output reconstructed_coords: {reconstructed_coords.shape}")
            self._debug_logged_forward = True
        return reconstructed_coords

class CrossAttentionBlock(nn.Module): # Provided but not used in Decoder2
    def __init__(self, q_dim, kv_dim, att_dim=64):
        super().__init__()
        self.scale_factor = att_dim**-0.5
        self.query_proj = nn.Linear(q_dim, att_dim, bias=False)
        self.key_proj = nn.Linear(kv_dim, att_dim, bias=False)
        self.value_proj = nn.Linear(kv_dim, att_dim, bias=False)
    def forward(self, q_input, k_input, v_input):
        Q = self.query_proj(q_input); K = self.key_proj(k_input); V = self.value_proj(v_input)
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale_factor
        attention_probs = F.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_probs, V)

class ProteinStateReconstructor2D(nn.Module):
    """Decoder for reconstructing coordinates from embeddings and conditioner."""
    _logged_forward_pass = False
    def __init__(self, atom_embedding_dim: int, num_atoms_total: int, conditioner_dim: int,
                 pooling_type: str = "blind", residue_segment_indices: Optional[List[List[int]]] = None,
                 primary_pool_output_size: Tuple[int, int] = (20, 4),
                 mlp_hidden_dim: int = 128, num_mlp_layers: int = 2,
                 use_secondary_pool: bool = False, secondary_pool_output_size: Optional[Tuple[int, int]] = None,
                 use_cross_attention: bool = False, cross_attention_type: str = "global", # Note: cross_attention not implemented
                 logger_instance: logging.Logger = logging.getLogger(__name__)):
        super().__init__()
        self.num_atoms_total = num_atoms_total
        self.atom_embedding_dim = atom_embedding_dim
        self.conditioner_dim = conditioner_dim
        self.logger = logger_instance
        self.pooling_type = pooling_type.lower()
        self.atom_segment_indices_list: List[torch.LongTensor] = []

        if self.pooling_type == "blind":
            self.atom_segment_indices_list.append(torch.arange(num_atoms_total, dtype=torch.long))
        elif self.pooling_type == "residue":
            if not residue_segment_indices: raise ValueError("Residue indices needed for 'residue' pooling.")
            self.atom_segment_indices_list = [torch.tensor(indices, dtype=torch.long) for indices in residue_segment_indices if indices]
            if not self.atom_segment_indices_list: raise ValueError("Empty residue segments for 'residue' pooling.")
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")

        self.num_segments_to_pool = len(self.atom_segment_indices_list)
        self.logger.info(f"Decoder2: PoolType='{self.pooling_type}', NumSegments={self.num_segments_to_pool}")

        self.segment_pooling_layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d(primary_pool_output_size) for _ in self.atom_segment_indices_list
        ])
        self.primary_pooled_feature_dim = primary_pool_output_size[0] * primary_pool_output_size[1]
        self.concatenated_after_primary_pool_dim = self.primary_pooled_feature_dim * self.num_segments_to_pool
        self.final_pooled_latent_dim = self.concatenated_after_primary_pool_dim
        self.secondary_global_pool = None

        if use_secondary_pool and secondary_pool_output_size and self.num_segments_to_pool > 0 :
            self.secondary_global_pool = nn.AdaptiveAvgPool2d(secondary_pool_output_size)
            self.final_pooled_latent_dim = secondary_pool_output_size[0] * secondary_pool_output_size[1]
            self.logger.info(f"Decoder2: Secondary pooling. FinalPooledDim={self.final_pooled_latent_dim}")
        else:
            self.logger.info(f"Decoder2: Primary pooling only. FinalPooledDim={self.final_pooled_latent_dim}")

        if use_cross_attention: self.logger.warning("Cross-attention not fully implemented in this Decoder2 version.")

        mlp_input_dim = self.conditioner_dim + self.final_pooled_latent_dim
        self.decoder_mlp = build_decoder_mlp(mlp_input_dim, 3, num_mlp_layers, mlp_hidden_dim)
        self.logger.info(f"Decoder2 MLP: In={mlp_input_dim}, Out=3, Layers={num_mlp_layers}, Hidden={mlp_hidden_dim}")

    def get_pooled_latent(self, atom_embeddings_batch: torch.Tensor) -> torch.Tensor:
        """Pools atom embeddings [B*N, E_atom] to global latent [B, E_pooled_final]."""
        if atom_embeddings_batch.ndim != 2 or atom_embeddings_batch.shape[0] % self.num_atoms_total != 0:
            raise ValueError(f"Bad atom_embeddings_batch shape: {atom_embeddings_batch.shape}")
        batch_size = atom_embeddings_batch.shape[0] // self.num_atoms_total
        atom_embeddings_reshaped = atom_embeddings_batch.view(batch_size, self.num_atoms_total, self.atom_embedding_dim)
        pooled_features_per_segment_list = []

        for i, atom_indices_for_segment in enumerate(self.atom_segment_indices_list):
            indices_on_device = atom_indices_for_segment.to(atom_embeddings_batch.device)
            if len(indices_on_device) == 0:
                pooled_features_per_segment_list.append(
                    torch.zeros(batch_size, self.primary_pooled_feature_dim,
                                device=atom_embeddings_batch.device, dtype=atom_embeddings_batch.dtype))
                continue
            segment_atom_embeddings = atom_embeddings_reshaped[:, indices_on_device, :]
            segment_atom_embeddings_for_pool = segment_atom_embeddings.unsqueeze(1) # [B, 1, N_seg_atoms, E_atom]
            pooled_segment_features = self.segment_pooling_layers[i](segment_atom_embeddings_for_pool)
            pooled_features_per_segment_list.append(pooled_segment_features.view(batch_size, -1))

        if not pooled_features_per_segment_list:
            return torch.zeros(batch_size, self.final_pooled_latent_dim,
                               device=atom_embeddings_batch.device, dtype=atom_embeddings_batch.dtype)

        concatenated_primary_pooled = torch.cat(pooled_features_per_segment_list, dim=1)

        if self.secondary_global_pool:
            # Input to 2nd pool depends on how primary pool outputs are arranged
            # Original code: l1_stack = torch.stack(seg_pooled, dim=1) # [B, num_segments, primary_dim]
            # self.glob_pool2(l1_stack.unsqueeze(1)).view(B,-1)
            # This implies if num_segments > 1, it pools over segments and primary_dim.
            # If blind (num_segments=1), concatenated_primary_pooled is [B, primary_dim]
            if self.pooling_type == "blind":
                # Reshape to [B, 1, 1, primary_dim] for pooling a single "image"
                reshaped_for_secondary = concatenated_primary_pooled.view(batch_size, 1, 1, -1)
            else: # Assume stacking for residue-wise, [B, 1, num_segments, primary_dim_per_segment]
                stacked_primary_pooled = torch.stack(pooled_features_per_segment_list, dim=1).unsqueeze(1)
                reshaped_for_secondary = stacked_primary_pooled
            final_pooled_latent = self.secondary_global_pool(reshaped_for_secondary).view(batch_size, -1)
        else:
            final_pooled_latent = concatenated_primary_pooled
        return final_pooled_latent

    def forward(self, atom_embeddings_flat: torch.Tensor, batch_indices: Optional[torch.Tensor],
                conditioner_per_atom_ref: torch.Tensor) -> torch.Tensor:
        """Predicts coords [B*N, 3] from embeddings [B*N, E_atom] and conditioner [N, E_cond]."""
        device = atom_embeddings_flat.device
        conditioner_on_device = conditioner_per_atom_ref.to(device)
        if atom_embeddings_flat.shape[1] != self.atom_embedding_dim or \
           conditioner_on_device.shape != (self.num_atoms_total, self.conditioner_dim):
            raise ValueError("Shape mismatch in Decoder2 input or conditioner.")
        num_total_atom_entries = atom_embeddings_flat.shape[0]
        if num_total_atom_entries == 0: return torch.empty(0,3,device=device) # Handle empty batch
        batch_size = num_total_atom_entries // self.num_atoms_total
        if num_total_atom_entries % self.num_atoms_total != 0:
             raise ValueError(f"Input atom_embeddings_flat dim 0 ({num_total_atom_entries}) not multiple of N_atoms ({self.num_atoms_total}).")


        global_pooled_latent_per_sample = self.get_pooled_latent(atom_embeddings_flat)
        global_pooled_latent_expanded = global_pooled_latent_per_sample.unsqueeze(1).expand(
            -1, self.num_atoms_total, -1) # [B, N, E_pooled_final]
        conditioner_expanded = conditioner_on_device.unsqueeze(0).expand(
            batch_size, -1, -1) # [B, N, E_cond]
        mlp_input_features = torch.cat([conditioner_expanded, global_pooled_latent_expanded], dim=-1)
        predicted_coords_flat = self.decoder_mlp(mlp_input_features.reshape(num_total_atom_entries, -1))

        if not ProteinStateReconstructor2D._logged_forward_pass and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"[Dec2 Fwd] Shapes: InEmb={atom_embeddings_flat.shape} CondRef={conditioner_per_atom_ref.shape} "
                             f"PoolOut={global_pooled_latent_per_sample.shape} MLPInFeatDim={mlp_input_features.shape[-1]} "
                             f"PredOut={predicted_coords_flat.shape}")
            ProteinStateReconstructor2D._logged_forward_pass = True
        return predicted_coords_flat
