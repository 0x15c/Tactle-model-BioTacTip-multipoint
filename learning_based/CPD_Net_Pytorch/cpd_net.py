import torch
import torch.nn as nn
import torch.nn.functional as F

class CPDNet(nn.Module):
    """CPD-Net model: predicts a displacement field to align a source point set to a target point set."""
    def __init__(self, point_dim=2, descriptor_dim=512):
        super(CPDNet, self).__init__()
        # Global shape descriptor encoder: 5-layer MLP (implemented as 1D conv layers) with dimensions 16,64,128,256,descriptor_dim:contentReference[oaicite:9]{index=9}.
        # Each conv1d layer with kernel_size=1 acts as a fully-connected layer applied to each point's features.
        # BatchNorm and ReLU are used after each layer (except the final output layer):contentReference[oaicite:10]{index=10}.
        input_channels = point_dim  # using (x,y) coordinates as input features for each point in 2D
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, descriptor_dim, kernel_size=1),
            nn.BatchNorm1d(descriptor_dim),
            nn.ReLU(inplace=True)
        )
        # PointMorph decoder: 3-layer MLP that takes [source_point + global_desc_source + global_desc_target] and outputs a 2D displacement:contentReference[oaicite:11]{index=11}.
        output_dim = point_dim  # 2 for 2D displacements
        self.point_morph = nn.Sequential(
            nn.Conv1d(descriptor_dim*2 + point_dim, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, output_dim, kernel_size=1)
            # No BatchNorm or ReLU on the final output layer (we want raw displacement values which can be positive or negative)
        )
    
    def forward(self, source_points, target_points):
        """
        Predict displacements to align source_points to target_points.
        source_points: Tensor of shape (B, N, 2)
        target_points: Tensor of shape (B, M, 2)
        Returns: Tensor of shape (B, N, 2) containing displacement vectors for each source point.
        """
        # Ensure batch dimension exists
        if source_points.dim() == 2:  # if input is (N, 2) for a single example
            source_points = source_points.unsqueeze(0)
        if target_points.dim() == 2:
            target_points = target_points.unsqueeze(0)
        B, N, _ = source_points.shape  # B = batch size, N = number of source points
        M = target_points.size(1)     # M = number of target points

        # Encode source and target point sets to global descriptors
        # Transpose to (B, features, N) for conv1d
        src_feat = source_points.transpose(1, 2)  # (B, 2, N)
        tgt_feat = target_points.transpose(1, 2)  # (B, 2, M)
        # Pass through the encoder MLP to get point-wise features
        src_feat = self.encoder(src_feat)  # (B, descriptor_dim, N)
        tgt_feat = self.encoder(tgt_feat)  # (B, descriptor_dim, M)
        # Global shape descriptors by max-pooling over points (order-invariant):contentReference[oaicite:12]{index=12}
        global_src = torch.max(src_feat, dim=2).values  # (B, descriptor_dim)
        global_tgt = torch.max(tgt_feat, dim=2).values  # (B, descriptor_dim)

        # Concatenate global descriptors with each source point coordinate:contentReference[oaicite:13]{index=13}:contentReference[oaicite:14]{index=14}
        # Expand global descriptors to shape (B, descriptor_dim, N) to concatenate with point features
        global_src_exp = global_src.unsqueeze(2).expand(B, global_src.size(1), N)  # (B, descriptor_dim, N)
        global_tgt_exp = global_tgt.unsqueeze(2).expand(B, global_tgt.size(1), N)  # (B, descriptor_dim, N)
        # Combine original source coordinates (as features) with both global descriptors
        src_coords = source_points.transpose(1, 2)  # (B, 2, N) original coords as features
        combined_feat = torch.cat([src_coords, global_src_exp, global_tgt_exp], dim=1)  # (B, 2+512+512, N) if descriptor_dim=512

        # Predict per-point displacement vectors (drifts) for the source set:contentReference[oaicite:15]{index=15}
        disp = self.point_morph(combined_feat)   # (B, 2, N)
        disp = disp.transpose(1, 2)  # (B, N, 2) to match input point ordering
        return disp

# Alignment loss: Chamfer Distance (CD) between two point sets:contentReference[oaicite:16]{index=16}.
def chamfer_distance(source_points, target_points, truncate=None):
    """
    Compute the bidirectional Chamfer Distance between source_points and target_points.
    If `truncate` is set (e.g., 0.1), uses a clipped Chamfer loss for robustness to outliers:contentReference[oaicite:17]{index=17}.
    source_points: (B, N, d) or (N, d)
    target_points: (B, M, d) or (M, d)
    Returns: scalar Chamfer loss.
    """
    # Ensure batch dimension
    if source_points.dim() == 2:
        source_points = source_points.unsqueeze(0)
    if target_points.dim() == 2:
        target_points = target_points.unsqueeze(0)
    # Compute pairwise distances between points
    # Use torch.cdist to get Euclidean distance matrix of shape (B, N, M)
    dist = torch.cdist(source_points, target_points, p=2.0)  # Euclidean distances
    dist2 = dist ** 2  # squared distances
    # For each source point, find the nearest target point distance
    min_src2tgt, _ = dist2.min(dim=2)  # (B, N)
    # For each target point, find the nearest source point distance
    min_tgt2src, _ = dist2.min(dim=1)  # (B, M)
    if truncate is not None:
        # Clip distances to not go below a minimum threshold (to limit influence of very close points or outliers)
        min_src2tgt = min_src2tgt.clamp(min=truncate)  # impose a floor of 'truncate' on distances:contentReference[oaicite:18]{index=18}
        min_tgt2src = min_tgt2src.clamp(min=truncate)
    # Average the two directional losses
    chamfer_loss = min_src2tgt.mean(dim=1) + min_tgt2src.mean(dim=1)  # per batch
    return chamfer_loss.mean()  # average over batch

# Example training loop usage:
# model = CPDNet()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# for epoch in range(num_epochs):
#     for source, target in dataloader:  # source, target: (B, N, 2) tensors
#         optimizer.zero_grad()
#         disp = model(source, target)           # predicted displacements
#         transformed_src = source + disp        # apply displacements to source points
#         loss = chamfer_distance(transformed_src, target, truncate=0.1)  # Chamfer alignment loss (clipped for outliers)
#         loss.backward()
#         optimizer.step()
