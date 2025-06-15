# Sequence Adjacency Matrix Predictor Script
# -----------------------------------------
# This script demonstrates a complete pipeline for predicting sequences of
# adjacency matrices (graph structures) from a single input adjacency matrix.
# Detailed Explanation for Thesis:
# ---------------------------------
# Overall Architecture:
# 1) Data Loading: We dynamically import a Python file defining `processX_adj`
#    and `controlX_Y_adj` tensors. Parsing is driven by naming conventions,
#    enabling flexible extension of examples without rewriting loader logic.
#    Each example consists of one process graph and multiple control graphs.
# 2) Dataset & Padding: Graphs vary in node count (size N). We compute a
#    global max_size across all graphs, then zero-pad each adjacency matrix
#    to [max_size x max_size]. Controls per example form a sequence tensor
#    [T x max_size x max_size], where T is the number of control graphs.
# 3) GCN Encoder:
#    - Input: Padded adjacency A ∈ ℝ^{B×M×M} for batch size B and max nodes M.
#    - Self-loops: A+I ensures each node sees itself.
#    - Normalization: Compute D^{-1/2} (A+I) D^{-1/2} for spectral
#      normalization, fostering stable message passing.
#    - Node feature: Degree (sum of each row) serves as initial scalar feature.
#    - Graph Convolution Layers:
#         h^{(1)} = ReLU(GCN1(deg))    # Linear(1→hidden) then message-passing
#         h^{(2)} = ReLU(An @ GCN2(h^{(1)}))  # Linear(hidden→d_model)
#      Two GCN layers produce final node embeddings of size d_model.
# 4) Transformer Decoder:
#    - Task: Auto-regressively predict each control adjacency.
#    - Tokens: Flatten each adjacency (M²) into a vector; prepend a zero start token.
#    - Embedding: Linear(M² → d_model) projects tokens into model dimension.
#    - Positional/causal masking: Standard upper-triangular mask prevents "future"
#      attention, enforcing autoregressive generation.
#    - Cross-Attention: Decoder layers attend to the encoder memory (node embeddings),
#      capturing structural context from the process graph.
# 5) Output Heads:
#    - Adjacency head: Linear(d_model → M²) + Sigmoid yields entries in [0,1].
#      During training, we use BCE loss against ground truth flattened adjacencies.
#    - EOS head: Linear(d_model → 1) + Logits indicates end-of-sequence at each step.
#      BCEWithLogits loss teaches model when to stop generating.
# 6) Training Loop:
#    - For each batch, obtain predictions [B×T×M×M] and EOS logits [B×T].
#    - Mask padding in control sequences so BCE only applies to real timesteps.
#    - Combine adjacency BCE loss + EOS BCE loss, backpropagate, and update parameters.
# 7) Inference:
#    - Given one process graph, we ask the decoder to generate a fixed number of
#      control graphs (T steps). We flatten and threshold at 0.5 to obtain binary
#      adjacency predictions.
# 8) Demonstration:
#    - We compare all original control graphs vs. predicted ones for example 1,
#      printing side-by-side matrices to assess performance.
#
# With this detailed understanding, one can trace every transformation from
# raw graph data through GCN embedding, self-attention decoding, to final control
# adjacency outputs. 

import sys  # for system exit on fatal errors
import argparse  # for parsing command-line arguments
import importlib.util  # for dynamically loading the dataset module
import os  # for filesystem checks
import matplotlib.pyplot as plt  # <-- Added for plotting training curves

# Ensure PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split 
except ImportError:
    sys.exit("Error: PyTorch is required. Install via 'pip install torch'.")


def parse_train_test(path):
    if not os.path.isfile(path):
        raise RuntimeError(f"Cannot open data file: {path}")

    # Dynamically import the dataset module
    spec = importlib.util.spec_from_file_location("dataset_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    examples_temp = []
    # Identify all process graphs by naming convention
    for name in dir(module):
        if name.startswith("process") and name.endswith("_adj"):
            idx_str = name[len("process"): -len("_adj")]
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            proc_mat = getattr(module, name)  # Tensor of shape [N, N]
           
              # Load process node features
            proc_feat_name = f"features_p{idx}"
            proc_feat = getattr(module, proc_feat_name, None)
            if proc_feat is None:
                raise ValueError(f"Missing features for process{idx}")
                # — Coerce to float tensor if needed —
                if not isinstance(proc_feat, torch.Tensor):
                    proc_feat = torch.tensor(proc_feat, dtype=torch.float)

            # Collect all matching controls for this process index
            ctrl_names = [c for c in dir(module)
                          if c.startswith(f"control{idx}_") and c.endswith("_adj")]
            # Sort controls by their numeric suffix to preserve order
            print(f"[DEBUG] Processing control group: {ctrl_names}")
            sorted_ctrls = sorted(
                ctrl_names,
                key=lambda nm: int(nm[len(f"control{idx}_"):-len("_adj")])
            )
            ctrls = [getattr(module, cname) for cname in sorted_ctrls]
            
             # Load control node features
            ctrl_feat_names = [
                #cname.replace("control", "features_tensor_c").replace("_adj", "")
                #for cname in sorted_ctrls
                f"feature_tensor_c{idx}_{n[len(f'control{idx}_'):-len('_adj')]}"
                for n in sorted_ctrls
            ]
            ctrl_feats = [getattr(module, fname, None) for fname in ctrl_feat_names]
            if any(cf is None for cf in ctrl_feats):
                raise ValueError(f"Missing control features for process{idx}")
                
            
            examples_temp.append((idx, proc_mat, proc_feat, ctrls, ctrl_feats))

    if not examples_temp:
        raise RuntimeError("No process/control definitions found.")

    # Sort by process index to maintain order 1,2,3...
    examples_temp.sort(key=lambda x: x[0])

    # Build final list of examples with debug output
    examples = []
    for idx, proc_mat, proc_feat, ctrls, ctrl_feats in examples_temp:
        examples.append({'process': proc_mat, 
                         'proc_feat': proc_feat, 
                         'controls': ctrls, 
                         'control_features': ctrl_feats})
        print(f"Loaded process{idx}_adj with {len(ctrls)} control matrices:")
        for i, (mat, feat) in enumerate(zip(ctrls, ctrl_feats), 1):
            print(f"  control{idx}_{i}_adj   shape {tuple(mat.shape)}")
    return examples


class ProcessControlDataset(Dataset):
    """
    PyTorch Dataset for paired (process, controls) adjacency matrices.
    - Pads all graphs to a uniform square size `max_size`.
    - Stacks multiple control graphs into a tensor of shape [T, max_size, max_size].
        - Returns:
        - process adjacency: [M, M]
        - process features: [M, F]
        - control adjacencies: [T, M, M]
        - control features: [T, M, F]
    """
    def __init__(self, examples, max_size=None):
        # Determine the maximum dimension across all exercises if not provided
        # --- NORMALIZATION SETUP: compute global mean/std on raw features --- <<<< ADD
        # concat all process features
        all_p = torch.cat([ex['proc_feat'] for ex in examples], dim=0)         # <<< ADD
        self.proc_feat_mean = all_p.mean(0)                                    # <<< ADD
        self.proc_feat_std  = all_p.std(0, unbiased=False).clamp(min=1e-6)     # <<< ADD
        # concat all control features
        all_c = torch.cat([cf for ex in examples for cf in ex['control_features']], dim=0)  # <<< ADD
        self.ctrl_feat_mean = all_c.mean(0)                                     # <<< ADD
        self.ctrl_feat_std  = all_c.std(0, unbiased=False).clamp(min=1e-6)      # <<< ADD
        # ---------------------------------------------------------------------        
        dims = []
        for ex in examples:
            dims.append(ex['process'].shape[0])
            for c in ex['controls']:
                dims.append(c.shape[0])
        self.max_size = max(dims) if max_size is None else max(max_size, max(dims))

        # Build padded data pairs
        self.data = []
        for ex in examples:
            # Pad process graph
            p = ex['process']
            if not isinstance(p, torch.Tensor):
                p = torch.tensor(p, dtype=torch.float)
            pad_p = torch.zeros(self.max_size, self.max_size)
            pad_p[:p.size(0), :p.size(1)] = p
            
            # process node features
            pf = ex['proc_feat']
            F_p = pf.size(1)
            pad_pf = torch.zeros(self.max_size, F_p)
            pad_pf[:pf.size(0), :pf.size(1)] = pf
            # --- NORMALIZE process features --- <<<< ADD
            pad_pf = (pad_pf - self.proc_feat_mean) / self.proc_feat_std      # <<< ADD

            # Pad and stack control graphs
            ctrl_tensors = []
            for c in ex['controls']:
                ct = c if isinstance(c, torch.Tensor) else torch.tensor(c, dtype=torch.float)
                pad_c = torch.zeros(self.max_size, self.max_size)
                pad_c[:ct.size(0), :ct.size(1)] = ct
                ctrl_tensors.append(pad_c)
            stacked_ctrl = torch.stack(ctrl_tensors)  # shape [T, max_size, max_size]
            
              # --- Control node feature sequence ---
            ctrl_feat_tensors = []
            for cf in ex['control_features']:
                F_c = cf.size(1)
                pad_cf = torch.zeros(self.max_size, F_c)
                pad_cf[:cf.size(0)] = cf
                ctrl_feat_tensors.append(pad_cf)
            ctrl_feat_stack = torch.stack(ctrl_feat_tensors)  # shape [T, M, F]
            # --- NORMALIZE control features ---
            ctrl_feat_stack = (ctrl_feat_stack - self.ctrl_feat_mean) / self.ctrl_feat_std  
            
            self.data.append((pad_p, pad_pf, stacked_ctrl, ctrl_feat_stack))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns tuple: (process_tensor [M,M], controls_tensor [T,M,M])
          # Returns:
        #   - process adjacency      [M, M]
        #   - process node features  [M, F_p]
        #   - control adjacencies    [T, M, M]
        #   - control node features  [T, M, F_c]
        # M = padded size, T = number of control graph in sequence, F_p, F_c: feature dimensions
        return self.data[idx]


def collate_fn(batch):
    """
    Custom collate for batching variable-length control sequences:
    Inputs:
        batch: list of tuples (proc [M,M], ctrls [T_i, M, M])
    Outputs:
        procs: Tensor [B, M, M]
        padded_ctrls: Tensor [B, T_max, M, M]
        mask: Bool Tensor [B, T_max] marking valid timesteps
    """
    procs_adj, proc_feat, ctrls_adj_seq, ctrl_feats_seq = zip(*batch)
    B = len(procs_adj)
    M = procs_adj[0].size(0)
    F_p = proc_feat[0].size(1)
    F_c = ctrl_feats_seq[0].size(2)

    # find longest sequence
    T_max = max(seq.size(0) for seq in ctrls_adj_seq)

    # initialize
    procs_adj = torch.stack(procs_adj)       # [B, M, M]
    proc_feat = torch.stack(proc_feat)       # [B, M, F_p]
    ctrl_adjs = torch.zeros(B, T_max, M, M)  # [B, T_max, M, M]
    ctrl_feats = torch.zeros(B, T_max, M, F_c)# [B, T_max, M, F_c]
    mask      = torch.zeros(B, T_max, dtype=torch.bool)  # <-- initialize mask
    
    # fill and mark mask
    for i in range(B):
        T = ctrls_adj_seq[i].size(0)
        ctrl_adjs[i, :T]  = ctrls_adj_seq[i]
        ctrl_feats[i, :T] = ctrl_feats_seq[i]
        mask[i, :T]       = True              # <-- **set** the valid positions

    return procs_adj, proc_feat, ctrl_adjs, ctrl_feats, mask


class SeqAdjPredictor(nn.Module):
    """
    Transformer-based sequence predictor with:
      - A 2-layer GCN encoder to embed the input process graph.
      - A multi-layer Transformer decoder to generate control graphs auto-regressively.
      - An EOS (end-of-sequence) head to learn when to stop (not used in fixed-length inference).

    Architecture Details:
      1) GCN Encoder:
         - Add self-loops, compute symmetric normalized adjacency A~ = D^-1/2 (A+I) D^-1/2.
         - Feature per node: degree of A~.
         - Two graph convolution layers: Linear -> ReLU -> A~ @ features.
         - Outputs node embeddings of size d_model for each of M nodes.

      2) Transformer Decoder:
         - At each decoding step t, input is flattened adjacency of previous step (or zero start vector).
         - Sequence of length T+1 fed through a learned Linear -> d_model.
         - Standard causal self-attention mask enforces auto-regressive property.
         - Attends over encoder memory (node embeddings) via cross-attention.

      3) Output Heads:
         - `out`: Linear(d_model -> M*M) + Sigmoid to predict next adjacency matrix.
         - `eos`: Linear(d_model -> 1) to predict probability next step is end-of-sequence.
    """
    def __init__(self, max_size, in_feats, d_model=512, nhead=8, layers=3,
                 gcn_layers=3, residual=False, pos_enc=False, node_feat_dim = 2):
        super().__init__()
        self.max_size = max_size
        self.d_model = d_model
        self.residual = residual # <- controls skip connections in GCN
        self.pos_enc = pos_enc # <- enable/disable positional encodings
        self.node_feat_dim = node_feat_dim
        
        # Build multiple GCN layers if requested
        self.gnns = nn.ModuleList()
        #in_feats = 1
        for _ in range(gcn_layers):
            self.gnns.append(nn.Linear(in_feats, d_model))
            in_feats = d_model
        # Transformer decoder stack
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        # Embedding for flattened adjacencies
        self.embed = nn.Linear(max_size*max_size, d_model)
        # Positional encoding lookup
        if self.pos_enc:
            self.pos_embedding = nn.Parameter(torch.randn(max_size*max_size, d_model)) # <-- add
        # Output heads
        self.out = nn.Linear(d_model, max_size*max_size)
        self.eos = nn.Linear(d_model, 1)
        self.node_feat_out = nn.Linear(d_model, node_feat_dim * max_size)
    
        self.log_var1 = nn.Parameter(torch.zeros(1))  # ← learn log σ₁² for feature-1
        self.log_var2 = nn.Parameter(torch.zeros(1))  # ← learn log σ₂² for feature-2

    def encode(self, A, X):
        B, M, _ = A.size()     
        I = torch.eye(M, device=A.device).unsqueeze(0).expand(B, M, M)          # add self loops to the adjacency matrix
        Ahat = A + I
        deg = Ahat.sum(-1)
        inv = deg.pow(-0.5); inv[inv == float('inf')] = 0
        D = torch.diag_embed(inv)
        An = D @ Ahat @ D
        #h = deg.unsqueeze(-1)
        h = X
        
        for layer in self.gnns:
            prev = h
            h = layer(h)
            h = An @ F.relu(h)
            if self.residual: # <-- add residual connection
                h = h + prev  # residual skip
        return h.transpose(0,1)

    def forward(self, proc_adj, proc_feat, tgt, tgt_mask=None, label_smoothing=0.0):
        B, T, M, _ = tgt.size()
        
        # Encode process graph
        mem = self.encode(proc_adj, proc_feat)
        
        # Prepare autoregressive input to decoder 
        sos = torch.zeros(B, M*M, device=proc_adj.device)
        seq = [sos] + [tgt[:,i].view(B,-1) for i in range(T)]
        seq = torch.stack(seq, dim=1)
        
        # Inject positional embeddings if enabled
        seq_emb = self.embed(seq)
        if self.pos_enc:
            #seq = seq + self.pos_embedding.unsqueeze(0)
            seq_len = seq_emb.size(1)
            seq_emb = seq_emb + self.pos_embedding[:seq_len].unsqueeze(0)
        emb = seq_emb.transpose(0, 1)   
        #emb = self.embed(seq).transpose(0,1)
        # Mask
        L = emb.size(0)
        causal = torch.triu(torch.full((L,L), float('-inf')), diagonal=1).to(proc_adj.device)
        
        # Transformer decoder
        dec = self.decoder(emb, mem, tgt_mask=causal)
        dec = dec[1:].transpose(0,1)
        # Adjacency prediction
        flat = torch.sigmoid(self.out(dec))
        adj = flat.view(B, T, M, M)
        # EOS prediction 
        eos_logits = self.eos(dec).squeeze(-1)
        
        node_feats_raw = self.node_feat_out(dec)  # [B, T, M * F_c]
        node_feats = node_feats_raw.view(B, T, M, self.node_feat_dim)  # [B, T, M, F_c]
        return adj, eos_logits, node_feats
       

    
def train_model(model, train_loader, val_loader, epochs, lr, device,
                weight_decay=0.0, clip_grad=0.0, label_smoothing=0.0,
                lambda_constraint=1.0, dataset=None): 
    
    """
    Standard training loop:
      - BCE loss on predicted adjacency entries vs. ground truth.
      - BCEWithLogits loss on EOS head to detect sequence end.
    Prints loss per epoch for monitoring.
    """
    # Optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
    mse = nn.MSELoss() 
    bce = nn.BCELoss()
    bcel = nn.BCEWithLogitsLoss()
    cel = nn.MSELoss()              # verify that this is the best loss criterion for unit and regulation?? No, entropy expects integer class labels
    
    
    train_losses, val_losses = [], []  # <-- storing both train & val losses
    total_correct = 0 
    total_count = 0 

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for batch_idx, (proc_adj, proc_feat, ctrl_adj, ctrl_feat, mask) in enumerate(train_loader, 1):
            proc_adj, proc_feat = proc_adj.to(device), proc_feat.to(device)
            ctrl_adj, ctrl_feat, mask = ctrl_adj.to(device), ctrl_feat.to(device), mask.to(device)
            # Label smoothing (optional)
            if label_smoothing > 0:
                eps = label_smoothing
                ctrl_adj_sm = ctrl_adj * (1 - eps) + 0.5 * eps
            else:
                ctrl_adj_sm = ctrl_adj
                
            pred_adj, eos_logits, pred_node_feat = model(proc_adj, proc_feat, ctrl_adj)

            # --- hard‐zero constraint penalty on A[0,0] ---
            # we want pred_adj[:,:,0,0] --> 0
            sl_constraint_loss = (pred_adj[:, :, 0, 0] ** 2).mean()
            pred_binary = (pred_adj > 0.5).float()

            # --- Constraint on range : all predicted types at last step ∈ [29, 54] ---
            types_pred = pred_node_feat[..., 0]  # shape [B, T, M]
            last_step_idx = mask.sum(dim=1) - 1  # shape [B]
            B = mask.size(0)
            last_types = types_pred[torch.arange(B), last_step_idx]  # shape [B, M]
            norm_min =(29.0 - dataset.ctrl_feat_mean[0]) / dataset.ctrl_feat_std[0]
            norm_max = (54.0 - dataset.ctrl_feat_mean[0]) / dataset.ctrl_feat_std[0]
            
            # Penalize anything below 29 or above 54 (after normalization)
            below_min = F.relu(norm_min - last_types)
            above_max = F.relu(last_types - norm_max)

            range_constraint_loss = (below_min + above_max).mean()
            lambda_range_constraint = 0.01  # start small and tune
            
            
            # Constraint: Node 0's type at last step == 29 (normalized target)
            norm_29 = (29.0 - dataset.ctrl_feat_mean[0]) / dataset.ctrl_feat_std[0]  # scalar

            node0_type_pred = torch.stack([
                pred_node_feat[b, last_step_idx[b], 0, 0] 
                for b in range(B)
            ])

            type_constraint_loss = ((node0_type_pred - norm_29) ** 2).mean()
            lambda_type_constraint = 0.1

            constraint_loss = sl_constraint_loss + type_constraint_loss * lambda_type_constraint + range_constraint_loss * lambda_range_constraint
            
            # mask for adjacency loss 
            mask_expand = mask.view(mask.size(0), mask.size(1), 1, 1).expand_as(ctrl_adj)
            l1 = bce(pred_adj[mask_expand], ctrl_adj_sm[mask_expand])
            
            # mask for node feature loss
            mask_node = mask.view(mask.size(0), mask.size(1), 1, 1).expand_as(ctrl_feat)
            # flatten out the masked features: [N_masked × F_c]
            pred_feats = pred_node_feat[mask_node].view(-1, model.node_feat_dim)
            true_feats = ctrl_feat[mask_node].view(-1, model.node_feat_dim)
            
            
            # two separate MSE losses
            l2_feat1 = mse(pred_feats[:, 0], true_feats[:, 0])
            l2_feat2 = mse(pred_feats[:, 1], true_feats[:, 1])
    
            # uncertainty‐weighted feature losses 
            loss_feat1 = 0.5 * torch.exp(-model.log_var1) * l2_feat1 \
                        + 0.5 * model.log_var1
            loss_feat2 = 0.5 * torch.exp(-model.log_var2) * l2_feat2 \
                        + 0.5 * model.log_var2           

            
            
            # EOS loss
            eos_target = torch.zeros_like(eos_logits)
            for i in range(mask.size(0)):
                eos_target[i, mask[i].sum().item() - 1] = 1
            l3 = bcel(eos_logits, eos_target)
            #loss = l1 + l2 + l3

            # now include weighted feature losses instead of raw l2
            loss = l1 + l3 + loss_feat1 + loss_feat2 \
                   + lambda_constraint * constraint_loss  

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad) 
            optimizer.step()

            correct = (pred_binary == ctrl_adj).float()
            total_correct += correct[mask_expand].sum().item()
            
            total_count += mask_expand.sum().item()
            total_loss += loss.item()
            #  Print every batch breakdown
            print(f"Epoch {epoch} Batch {batch_idx}: "
                  f"adj_loss={l1.item():.4f}, "
                  f"feat1_loss={l2_feat1.item():.4f}, "
                  f"feat2_loss={l2_feat2.item():.4f}, "
                  f"eos_loss={l3.item():.4f}"
                  f", constraint_loss={constraint_loss.item():.4f}")            

        avg_train = total_loss / len(train_loader)
        train_losses.append(avg_train)
        print(f"Epoch {epoch} — Train Accuracy (Adjacency) ≈ {total_correct / total_count:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for proc_adj, proc_feat, ctrl_adj, ctrl_feat, mask in val_loader:
                proc_adj, proc_feat = proc_adj.to(device), proc_feat.to(device)
                ctrl_adj, ctrl_feat, mask = ctrl_adj.to(device), ctrl_feat.to(device), mask.to(device)

                pred_adj, eos_logits, pred_node_feat = model(proc_adj, proc_feat, ctrl_adj)

                mask_expand = mask.view(mask.size(0), mask.size(1), 1, 1).expand_as(ctrl_adj)
                l1 = bce(pred_adj[mask_expand], ctrl_adj[mask_expand])

                mask_node = mask.view(mask.size(0), mask.size(1), 1, 1).expand_as(ctrl_feat)
                l2 = cel(pred_node_feat[mask_node], ctrl_feat[mask_node])

                eos_target = torch.zeros_like(eos_logits)
                
                for i in range(mask.size(0)):
                    eos_target[i, mask[i].sum().item() - 1] = 1
                l3 = bcel(eos_logits, eos_target)

                val_loss += (l1 + l2 + l3).item()

            avg_val = val_loss / len(val_loader)
            val_losses.append(avg_val)

            print(f"Epoch {epoch}/{epochs} — Train Loss: {avg_train:.4f} — Val Loss: {avg_val:.4f}")
            overall_train_acc = total_correct / total_count
            

    return train_losses, val_losses, overall_train_acc
                


def infer(model, proc_adj, proc_feat, max_steps, dataset):
    """
    Generate a fixed number of control adjacency matrices auto-regressively.
    Also returns predicted node features and eos scores.
    """
    model.eval()
    with torch.no_grad():
        proc_adj = proc_adj.unsqueeze(0).to(next(model.parameters()).device)
        proc_feat = proc_feat.unsqueeze(0).to(proc_adj.device)
        mem = model.encode(proc_adj, proc_feat)

        tok = [torch.zeros(1, model.max_size * model.max_size, device=proc_adj.device)]
        outs = []
        feats = []
        eos_scores = []

        for _ in range(max_steps):
            seq = torch.stack(tok, dim=1)  # [1, T, M*M]
            seq_emb = model.embed(seq)
            if model.pos_enc:
                seq_len = seq_emb.size(1)
                seq_emb = seq_emb + model.pos_embedding[:seq_len].unsqueeze(0)
            emb = seq_emb.transpose(0, 1)
            L = emb.size(0)
            
            causal_mask = torch.triu(torch.full((L, L), float('-inf')), diagonal=1).to(proc_adj.device)
            dec = model.decoder(emb, mem, tgt_mask=causal_mask)[-1]  # [1, d_model]
            flat = torch.sigmoid(model.out(dec)).view(1, model.max_size, model.max_size)
            outs.append(flat[0])
            feat_raw = model.node_feat_out(dec).view(1, model.max_size, model.node_feat_dim)
            feats.append(feat_raw[0])
            eos_score = torch.sigmoid(model.eos(dec)).squeeze(-1)
            eos_scores.append(eos_score.item())

            tok.append(flat.view(1, -1))

        return torch.stack(outs), torch.stack(feats), eos_scores



# +---------------------------------------------+
# |                   MAIN                      |
# +---------------------------------------------+
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Dataset_general_features.py')
    parser.add_argument('--max_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=16)  
    parser.add_argument('--epochs', type=int, default=300)         
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--gcn_layers', type=int, default=6)  
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--pos_enc', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-3)  
    parser.add_argument('--clip_grad', type=float, default=1.0)    
    parser.add_argument('--lambda_constraint', type=float, default=1.0,
                    help='weight for enforcing A[0,0]==0 penalty')
    args = parser.parse_args()
    λ_constr = args.lambda_constraint
    
    # Load dataset
    examples = parse_train_test(args.data)
    dataset = ProcessControlDataset(examples, args.max_size)
    # Split train/validation
    val_size = max(1, int(0.1 * len(dataset)))     # <-- 10% validation split
    
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])  # <-- using random_split
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    in_feats = dataset[0][1].shape[-1]  # shape: [M, F_p] → get F_p
    model = SeqAdjPredictor(dataset.max_size,
                            in_feats=in_feats,
                            d_model=512,
                            #d_model=args.d_model,
                            nhead=8,
                            layers=4,
                            #layers=args.transformer_layers,  # <--- updated
                            #gcn_layers=args.gcn_layers,
                            gcn_layers=6,
                            residual=args.residual,
                            pos_enc=args.pos_enc).to(device)

    # Train and capture histories
    train_losses, val_losses, overall_train_acc = train_model(
        model,
        train_loader,
        val_loader,
        args.epochs,
        args.lr,
        device,
        weight_decay=args.weight_decay,  # <-- passing weight_decay
        clip_grad=args.clip_grad,        # <-- passing clip_grad
        label_smoothing=args.label_smoothing,
        lambda_constraint=args.lambda_constraint,
        dataset=dataset
    )
    print(f"\nFinal Overall Training Accuracy (Adjacency): {overall_train_acc:.4f}")

    

# -------------------------
    # Inference on first x examples
    # -------------------------
    num_display = 5  
    for ex_idx in range(num_display):  
        proc_adj, proc_feat, _, _ = dataset[ex_idx]
        raw_ctrls = examples[ex_idx]['controls']
        n_ctrl = len(raw_ctrls)
        preds, pred_feats, eos_scores = infer(model, proc_adj, proc_feat, max_steps=n_ctrl, dataset=dataset)
        binarized = (preds > 0.5).int()

        print(f"\nExample {ex_idx+1} — Found {n_ctrl} controls; generated {binarized.size(0)} predictions.")
        for j in range(n_ctrl):
            orig = raw_ctrls[j].int().numpy()
            pred = binarized[j][:orig.shape[0], :orig.shape[1]].cpu().numpy()
            feat_norm = pred_feats[j][:orig.shape[0]].cpu()                         
            feat_denorm = feat_norm * dataset.ctrl_feat_std + dataset.ctrl_feat_mean 
            feat_denorm = feat_denorm.numpy()                                       
            feat = pred_feats[j][:orig.shape[0]].cpu().numpy()
            print(f"-- Control {j+1} --")
            print("Original:")
            print(orig)
            print("Predicted:")
            print(pred)
            print("Predicted Node Features (type + regulation):")
            print(feat.round(2))
            print("Predicted Node Features (denormalized):")                        
            print(feat_denorm.round(2))                                                        

if __name__ == '__main__':
    main()
