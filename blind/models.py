import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys # For initial HNO debug flush
from typing import Dict, List, Optional, Tuple, Any # Added Any for ProteinStateReconstructor2D logger

from torch_geometric.nn import ChebConv
from math_utils import build_decoder_mlp # Assuming math_utils.py contains build_decoder_mlp

logger = logging.getLogger(__name__)


class HNO(nn.Module):
    """
    HNO (Hierarchical Neural Operator) Encoder based on ChebConv layers.
    This version matches the paper's description: 4 ChebConv layers,
    LeakyReLU for first two, ReLU for third, BatchNorm after each,
    and L2 normalization on final atom embeddings.
    The 'mlpRep' is a linear layer for direct coordinate reconstruction if used for pre-training.
    """
    def __init__(self, hidden_dim: int, K_cheb_order: int):
        super().__init__()
        self._debug_logged_forward = False # For one-time debug logging in forward
        self._debug_logged_repres = False # For one-time debug logging in forward_representation

        logger.debug(f"Initializing HNO Encoder with hidden_dim={hidden_dim}, K_cheb_order={K_cheb_order}")
        sys.stdout.flush() # As in original for early debug

        # Input dimension is 3 (x, y, z coordinates)
        self.conv1 = ChebConv(in_channels=3, out_channels=hidden_dim, K=K_cheb_order)
        self.bano1 = nn.BatchNorm1d(num_features=hidden_dim)

        self.conv2 = ChebConv(in_channels=hidden_dim, out_channels=hidden_dim, K=K_cheb_order)
        self.bano2 = nn.BatchNorm1d(num_features=hidden_dim)

        self.conv3 = ChebConv(in_channels=hidden_dim, out_channels=hidden_dim, K=K_cheb_order)
        self.bano3 = nn.BatchNorm1d(num_features=hidden_dim)

        self.conv4 = ChebConv(in_channels=hidden_dim, out_channels=hidden_dim, K=K_cheb_order)
        # No BatchNorm after the final ChebConv before L2 normalization, as per paper's described flow for Z.

        # Final linear layer for direct coordinate prediction (e.g., for pre-training the encoder)
        # This is effectively the MLP_HNO from the paper's loss eq. (E.1) if this HNO is trained standalone.
        self.mlpRep = nn.Linear(hidden_dim, 3)


    def forward_representation(self, x: torch.Tensor, edge_index: torch.Tensor,
                               log_debug_shapes: bool = False) -> torch.Tensor:
        """
        Computes the latent atom-wise embeddings Z.
        Output shape: [N_atoms_in_batch, hidden_dim]
        """
        if log_debug_shapes and not self._debug_logged_repres:
            logger.debug(f"[HNO.rep] Input x shape: {x.shape}, edge_index shape: {edge_index.shape}")

        x = x.float() # Ensure input is float

        x = F.leaky_relu(self.bano1(self.conv1(x, edge_index)))
        if log_debug_shapes and not self._debug_logged_repres:
            logger.debug(f"[HNO.rep] After conv1+bano1+act: {x.shape}")

        x = F.leaky_relu(self.bano2(self.conv2(x, edge_index)))
        if log_debug_shapes and not self._debug_logged_repres:
            logger.debug(f"[HNO.rep] After conv2+bano2+act: {x.shape}")

        x = F.relu(self.bano3(self.conv3(x, edge_index))) # ReLU for the third activation
        if log_debug_shapes and not self._debug_logged_repres:
            logger.debug(f"[HNO.rep] After conv3+bano3+act: {x.shape}")

        x = self.conv4(x, edge_index) # Final graph convolution
        if log_debug_shapes and not self._debug_logged_repres:
            logger.debug(f"[HNO.rep] After conv4: {x.shape}")

        # L2 normalization per atom embedding, as described in paper (Sec 3.2)
        z_embeddings = F.normalize(x, p=2.0, dim=1)
        if log_debug_shapes and not self._debug_logged_repres:
            logger.debug(f"[HNO.rep] Output Z embeddings shape (after L2 norm): {z_embeddings.shape}")
            self._debug_logged_repres = True
        return z_embeddings

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                log_debug_shapes: bool = False) -> torch.Tensor:
        """
        Full forward pass including the final MLP for coordinate reconstruction (e.g., for pre-training).
        Output shape: [N_atoms_in_batch, 3]
        """
        # Get the latent embeddings Z
        z_embeddings = self.forward_representation(x, edge_index, log_debug_shapes=log_debug_shapes)

        # Pass Z through the final MLP to get reconstructed coordinates
        reconstructed_coords = self.mlpRep(z_embeddings)

        if log_debug_shapes and not self._debug_logged_forward: # Use a separate flag for this method
            logger.debug(f"[HNO.fwd] Output reconstructed_coords shape: {reconstructed_coords.shape}")
            self._debug_logged_forward = True
        return reconstructed_coords


class CrossAttentionBlock(nn.Module): # As provided, though not used in ProteinStateReconstructor2D
    def __init__(self, q_dim, kv_dim, att_dim=64):
        super().__init__()
        self.scale_factor = att_dim**-0.5
        self.query_proj = nn.Linear(q_dim, att_dim, bias=False)
        self.key_proj = nn.Linear(kv_dim, att_dim, bias=False)
        self.value_proj = nn.Linear(kv_dim, att_dim, bias=False)

    def forward(self, q_input, k_input, v_input):
        Q = self.query_proj(q_input)
        K = self.key_proj(k_input)
        V = self.value_proj(v_input)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale_factor
        attention_probs = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, V)
        return output


class ProteinStateReconstructor2D(nn.Module):
    """
    Decoder model that reconstructs full atom coordinates from atom-wise latent embeddings (from HNO)
    and a conditioner. Implements pooling strategies as described in the paper.
    This version is for 'blind' pooling.
    """
    _logged_forward_pass = False # Class variable for one-time logging

    def __init__(self,
                 atom_embedding_dim: int,    # Dimensionality of input atom embeddings (d_z from HNO)
                 num_atoms_total: int,       # Total number of atoms in the protein structure (N)
                 conditioner_dim: int,     # Dimensionality of the conditioner C
                 pooling_type: str = "blind",# 'blind' or 'residue' (residue not fully implemented here)
                 residue_segment_indices: Optional[List[List[int]]] = None, # For 'residue' pooling
                 primary_pool_output_size: Tuple[int, int] = (20, 4), # (H, W) for AdaptiveAvgPool2d
                 mlp_hidden_dim: int = 128,
                 num_mlp_layers: int = 2,    # Total layers in MLP (e.g. 2 => In -> Hidden -> Out)
                 use_secondary_pool: bool = False,
                 secondary_pool_output_size: Optional[Tuple[int, int]] = None,
                 use_cross_attention: bool = False, # Not implemented in detail here
                 cross_attention_type: str = "global",
                 logger_instance: logging.Logger = logging.getLogger(__name__)):
        super().__init__()
        self.num_atoms_total = num_atoms_total
        self.atom_embedding_dim = atom_embedding_dim # d_z
        self.conditioner_dim = conditioner_dim     # d_c
        self.logger = logger_instance
        self.pooling_type = pooling_type.lower()

        self.atom_segment_indices_list: List[torch.LongTensor] = [] # For pooling different parts of protein

        if self.pooling_type == "blind":
            # For blind pooling, the "segment" is all atoms
            self.atom_segment_indices_list.append(torch.arange(num_atoms_total, dtype=torch.long))
        elif self.pooling_type == "residue":
            if not residue_segment_indices:
                raise ValueError("Residue indices must be provided for 'residue' pooling type.")
            self.atom_segment_indices_list = [
                torch.tensor(indices, dtype=torch.long) for indices in residue_segment_indices if indices
            ]
            if not self.atom_segment_indices_list:
                raise ValueError("Empty residue segments provided for 'residue' pooling.")
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}. Supported: 'blind', 'residue'.")

        self.num_segments_to_pool = len(self.atom_segment_indices_list)
        self.logger.info(f"Decoder2 initialized with pooling_type='{self.pooling_type}', "
                         f"Number of segments to pool: {self.num_segments_to_pool}")

        # Primary pooling layer(s) for each segment
        self.segment_pooling_layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d(primary_pool_output_size) for _ in self.atom_segment_indices_list
        ])
        self.primary_pooled_feature_dim = primary_pool_output_size[0] * primary_pool_output_size[1]
        
        # Concatenated dimension after primary pooling over all segments
        # If blind, num_segments_to_pool = 1, so this is just primary_pooled_feature_dim
        self.concatenated_after_primary_pool_dim = self.primary_pooled_feature_dim * self.num_segments_to_pool

        self.secondary_global_pool = None
        self.final_pooled_latent_dim = self.concatenated_after_primary_pool_dim

        if use_secondary_pool and secondary_pool_output_size and self.num_segments_to_pool > 0:
            # This secondary pool operates on the *concatenated* (or stacked) primary pooled features
            self.secondary_global_pool = nn.AdaptiveAvgPool2d(secondary_pool_output_size)
            self.final_pooled_latent_dim = secondary_pool_output_size[0] * secondary_pool_output_size[1]
            self.logger.info(f"Decoder2: Secondary pooling enabled. Final pooled latent dim = {self.final_pooled_latent_dim}")
        else:
            self.logger.info(f"Decoder2: Only primary pooling. Concatenated primary pooled dim = {self.concatenated_after_primary_pool_dim}")


        # Cross-attention (placeholder, not fully integrated based on provided snippet)
        self.cross_attention_layer = None
        if use_cross_attention:
            self.logger.warning("Cross-attention is marked for use but its detailed implementation in ProteinStateReconstructor2D is omitted in the provided snippet.")
            # self.cross_attention_layer = CrossAttentionBlock(...)

        # MLP to predict coordinates
        # Input to MLP: conditioner_dim (per atom) + final_pooled_latent_dim (global, tiled per atom)
        mlp_input_dim = self.conditioner_dim + self.final_pooled_latent_dim
        self.decoder_mlp = build_decoder_mlp(
            input_dim=mlp_input_dim,
            output_dim=3, # Predicting 3D coordinates
            num_hidden_layers=num_mlp_layers,
            hidden_layer_dim=mlp_hidden_dim
        )
        self.logger.info(f"Decoder2 MLP: InputDim={mlp_input_dim}, OutputDim=3, "
                         f"NumLayers={num_mlp_layers}, HiddenDim={mlp_hidden_dim}")

    def get_pooled_latent(self, atom_embeddings_batch: torch.Tensor) -> torch.Tensor:
        """
        Pools atom-wise embeddings to get a global latent vector per sample in the batch.
        Input: atom_embeddings_batch of shape [BatchSize * NumAtoms, AtomEmbeddingDim]
        Output: pooled_latent_batch of shape [BatchSize, final_pooled_latent_dim]
        """
        if atom_embeddings_batch.ndim != 2 or atom_embeddings_batch.shape[0] % self.num_atoms_total != 0:
            raise ValueError(f"Input atom_embeddings_batch has incorrect shape: {atom_embeddings_batch.shape}. "
                             f"Expected [B*N, E] where N={self.num_atoms_total}.")

        batch_size = atom_embeddings_batch.shape[0] // self.num_atoms_total
        # Reshape to [BatchSize, NumAtoms, AtomEmbeddingDim] for easier atom selection per segment
        atom_embeddings_reshaped = atom_embeddings_batch.view(batch_size, self.num_atoms_total, self.atom_embedding_dim)

        pooled_features_per_segment_list = []
        for i, atom_indices_for_segment in enumerate(self.atom_segment_indices_list):
            indices_on_device = atom_indices_for_segment.to(atom_embeddings_batch.device)

            if len(indices_on_device) == 0: # Handle empty segments if they occur
                # Append zeros of the correct primary pooled dimension
                pooled_features_per_segment_list.append(
                    torch.zeros(batch_size, self.primary_pooled_feature_dim,
                                device=atom_embeddings_batch.device, dtype=atom_embeddings_batch.dtype)
                )
                continue

            # Select embeddings for the current segment: [BatchSize, NumAtomsInSegment, AtomEmbeddingDim]
            segment_atom_embeddings = atom_embeddings_reshaped[:, indices_on_device, :]

            # Add a channel dimension for 2D pooling: [BatchSize, 1, NumAtomsInSegment, AtomEmbeddingDim]
            # The paper mentions "reshaping Z(t) as an image-like tensor N x dz"
            # This implies pooling operates over N (atoms) and dz (features)
            # AdaptiveAvgPool2d expects (B, C_in, H_in, W_in) -> pools H_in, W_in
            # Here, we want to pool across atoms in the segment (NumAtomsInSegment) and features (AtomEmbeddingDim)
            # The original script uses: seg_x = x_r[:,idx_dev,:].unsqueeze(1); pool_f=self.seg_pools[i](seg_x).view(B,-1);
            # This means C_in = 1, H_in = NumAtomsInSegment, W_in = AtomEmbeddingDim
            segment_atom_embeddings_for_pool = segment_atom_embeddings.unsqueeze(1)

            pooled_segment_features = self.segment_pooling_layers[i](segment_atom_embeddings_for_pool) # [B, 1, H_out, W_out]
            pooled_features_per_segment_list.append(pooled_segment_features.view(batch_size, -1)) # Flatten to [B, primary_pooled_dim]

        if not pooled_features_per_segment_list: # Should not happen if constructor validates
            return torch.zeros(batch_size, self.final_pooled_latent_dim,
                               device=atom_embeddings_batch.device, dtype=atom_embeddings_batch.dtype)

        # Concatenate features from all segments: [BatchSize, NumSegments * primary_pooled_dim]
        # If blind pooling, NumSegments = 1, so this is just [B, primary_pooled_dim]
        concatenated_primary_pooled_features = torch.cat(pooled_features_per_segment_list, dim=1) # [B, concatenated_after_primary_pool_dim]


        if self.secondary_global_pool:
            # Secondary pool expects [B, C_in, H_in, W_in].
            # Here, H_in could be NumSegments, W_in could be primary_pooled_feature_dim if stacked.
            # The original code used: l1_stack = torch.stack(seg_pooled, dim=1) # [B, num_segments, primary_dim]
            # self.glob_pool2(l1_stack.unsqueeze(1)).view(B,-1)
            # This implies C_in = 1, H_in = NumSegments, W_in = primary_pooled_feature_dim
            # If we've concatenated, then C_in=1, H_in=1, W_in = concatenated_after_primary_pool_dim
            # For blind pooling (num_segments=1), the original stack becomes [B,1,primary_dim]
            # So l1_stack.unsqueeze(1) would be [B,1,1,primary_dim] for pooling.

            if self.pooling_type == "blind": # num_segments is 1
                # Input to secondary_pool: [B, 1, 1, primary_pooled_feature_dim]
                reshaped_for_secondary_pool = concatenated_primary_pooled_features.view(batch_size, 1, 1, -1)
            else: # For residue pooling, treat num_segments as one dimension, primary_pooled_feature_dim as another
                # Input to secondary_pool: [B, 1, num_segments, primary_pooled_feature_dim]
                reshaped_for_secondary_pool = torch.stack(pooled_features_per_segment_list, dim=1).unsqueeze(1)


            final_pooled_latent = self.secondary_global_pool(reshaped_for_secondary_pool).view(batch_size, -1)
        else:
            final_pooled_latent = concatenated_primary_pooled_features

        return final_pooled_latent


    def forward(self, atom_embeddings_flat: torch.Tensor, # [BatchSize * NumAtoms, AtomEmbeddingDim]
                batch_indices: Optional[torch.Tensor],      # PyG batch vector, not directly used here if B*N input
                conditioner_per_atom: torch.Tensor          # [NumAtoms, ConditionerDim] - same for all in batch
                ) -> torch.Tensor:
        """
        Predicts coordinates from atom embeddings and a conditioner.
        Output shape: [BatchSize * NumAtoms, 3]
        """
        device = atom_embeddings_flat.device
        conditioner_on_device = conditioner_per_atom.to(device)

        # Validate input shapes
        if atom_embeddings_flat.shape[1] != self.atom_embedding_dim:
            raise ValueError(f"Atom embeddings dimension mismatch. Expected {self.atom_embedding_dim}, "
                             f"got {atom_embeddings_flat.shape[1]}.")
        if conditioner_on_device.shape != (self.num_atoms_total, self.conditioner_dim):
            raise ValueError(f"Conditioner dimension mismatch. Expected ({self.num_atoms_total}, {self.conditioner_dim}), "
                             f"got {conditioner_on_device.shape}.")

        num_total_atom_entries = atom_embeddings_flat.shape[0]
        if num_total_atom_entries % self.num_atoms_total != 0:
            # Try to infer batch size if batch_indices are provided (typical in PyG loaders)
            if batch_indices is not None:
                batch_size = batch_indices.max().item() + 1
                if batch_size * self.num_atoms_total != num_total_atom_entries:
                    raise ValueError(f"Input atom_embeddings_flat shape [B*N,E] inconsistent. "
                                     f"B*N = {num_total_atom_entries} but N={self.num_atoms_total}. Inferred B from batch_indices={batch_size}")
            else: # Cannot infer batch size
                 raise ValueError(f"Input atom_embeddings_flat shape [B*N,E] inconsistent. "
                                 f"B*N = {num_total_atom_entries} but N={self.num_atoms_total}. Batch indices not provided.")
        else:
            batch_size = num_total_atom_entries // self.num_atoms_total

        # 1. Get the global pooled latent vector for each sample in the batch
        # Input: [B*N, E_atom], Output: [B, E_pooled_final]
        global_pooled_latent_per_sample = self.get_pooled_latent(atom_embeddings_flat)

        # 2. Expand/Tile the global pooled latent to be per-atom
        # Output: [B, N, E_pooled_final]
        global_pooled_latent_expanded = global_pooled_latent_per_sample.unsqueeze(1).expand(
            -1, self.num_atoms_total, -1
        )

        # 3. Expand/Tile the conditioner to match batch size
        # Input: [N, E_cond], Output: [B, N, E_cond]
        conditioner_expanded = conditioner_on_device.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # 4. Concatenate per-atom conditioner and expanded global pooled latent
        # Output: [B, N, E_cond + E_pooled_final]
        mlp_input_features_per_atom = torch.cat(
            [conditioner_expanded, global_pooled_latent_expanded], dim=-1
        )

        # 5. Flatten for MLP and predict coordinates
        # Input to MLP: [B*N, E_cond + E_pooled_final]
        # Output from MLP: [B*N, 3]
        predicted_coords_flat = self.decoder_mlp(
            mlp_input_features_per_atom.view(num_total_atom_entries, -1)
        )

        if not ProteinStateReconstructor2D._logged_forward_pass and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"[Decoder2 Forward Pass Shapes - PoolType: {self.pooling_type}]")
            self.logger.debug(f"  Input atom_embeddings_flat: {atom_embeddings_flat.shape}")
            self.logger.debug(f"  Input conditioner_per_atom: {conditioner_per_atom.shape}")
            self.logger.debug(f"  Inferred Batch Size: {batch_size}")
            self.logger.debug(f"  Global pooled latent (per sample): {global_pooled_latent_per_sample.shape}")
            self.logger.debug(f"  MLP input feature dim (per atom): {mlp_input_features_per_atom.shape[-1]}")
            self.logger.debug(f"  Output predicted_coords_flat: {predicted_coords_flat.shape}")
            ProteinStateReconstructor2D._logged_forward_pass = True

        return predicted_coords_flat
