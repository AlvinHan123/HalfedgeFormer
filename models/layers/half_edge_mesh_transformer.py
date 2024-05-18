import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class HalfEdgeMeshTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, dim_feedforward=512, num_layers=6, dropout=0.1):
        super(HalfEdgeMeshTransformer, self).__init__()
        self.projected_dim = num_heads * (
                    (in_channels + num_heads - 1) // num_heads)  # Smallest multiple of num_heads >= input_dim

        # Linear projection layer to ensure input_dim is divisible by num_heads
        self.projection = nn.Linear(in_channels, self.projected_dim)

        # Encoder layer and Transformer Encoder
        self.encoder_layer = TransformerEncoderLayer(d_model=self.projected_dim, nhead=num_heads,
                                                     dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Linear layer to project back to the desired output dimension
        self.fc = nn.Linear(self.projected_dim, out_channels)

    def forward(self, half_edge_features, meshes):
        half_edge_features = half_edge_features.squeeze(-1)
        number_of_half_edges_in_features = half_edge_features.shape[2]
        device = half_edge_features.device
        batch_half_edge_neighborhoods = torch.cat(
            [self.get_prepared_half_edge_neighborhoods_from_mesh(i, number_of_half_edges_in_features, device) for i in
             meshes], 0)

        features_of_neighborhoods = self.__gather_neighborhood_features(half_edge_features,
                                                                        batch_half_edge_neighborhoods)
        features_of_neighborhoods = self.projection(features_of_neighborhoods)

        features_of_neighborhoods = features_of_neighborhoods.permute(1, 0,
                                                                      2)  # Transformer expects (sequence_length, batch_size, input_dim)
        transformed_features = self.transformer_encoder(features_of_neighborhoods)
        transformed_features = transformed_features.permute(1, 0, 2)  # Back to (batch_size, sequence_length, input_dim)

        transformed_features = transformed_features.mean(dim=1)  # Global average pooling

        half_edge_features = self.fc(transformed_features)

        return half_edge_features.unsqueeze(-1)

    def get_prepared_half_edge_neighborhoods_from_mesh(self, mesh, padding_target, device):
        half_edge_neighborhoods = torch.tensor(mesh.half_edge_neighborhoods, dtype=torch.float32, device=device)
        half_edge_neighborhoods = half_edge_neighborhoods.requires_grad_()
        half_edge_ids = torch.arange(mesh.half_edge_count, dtype=torch.float32, device=device).unsqueeze(1)
        half_edge_neighborhoods = torch.cat((half_edge_ids, half_edge_neighborhoods), dim=1)
        padding = (0, 0, 0, padding_target - mesh.half_edge_count)
        half_edge_neighborhoods = F.pad(half_edge_neighborhoods, pad=padding, mode="constant", value=0)
        half_edge_neighborhoods = half_edge_neighborhoods.unsqueeze(0)
        return half_edge_neighborhoods

    def __gather_neighborhood_features(self, half_edge_features, half_edge_neighborhoods):
        num_batches, _, num_half_edges = half_edge_features.shape
        nbh_size_plus_one = half_edge_neighborhoods.shape[2]
        half_edge_neighborhoods = self.__prepare_half_edge_indices(half_edge_neighborhoods)
        half_edge_features = self.__prepare_half_edge_features(half_edge_features)
        features_of_neighborhoods = torch.index_select(half_edge_features, dim=0, index=half_edge_neighborhoods)
        features_of_neighborhoods = features_of_neighborhoods.view(num_batches, num_half_edges, nbh_size_plus_one, -1)
        features_of_neighborhoods = features_of_neighborhoods.permute(0, 2, 1, 3)
        return features_of_neighborhoods

    def __prepare_half_edge_indices(self, half_edge_neighborhoods):
        half_edge_neighborhoods = half_edge_neighborhoods + 1
        number_of_meshes_in_batch, number_of_half_edges_in_mesh, nbh_size_plus_one = half_edge_neighborhoods.shape
        for mesh_number in range(number_of_meshes_in_batch):
            half_edge_neighborhoods[mesh_number] += (number_of_half_edges_in_mesh + 1) * mesh_number
        half_edge_neighborhoods = half_edge_neighborhoods.view(-1).long()
        return half_edge_neighborhoods

    def __prepare_half_edge_features(self, half_edge_features):
        num_batches, num_channels, _ = half_edge_features.shape
        device = half_edge_features.device
        padding = torch.zeros((num_batches, num_channels, 1), requires_grad=True, device=device)
        half_edge_features = torch.cat((padding, half_edge_features), dim=2)
        num_half_edges_new = half_edge_features.shape[2]
        half_edge_features = half_edge_features.permute(0, 2, 1).contiguous()
        half_edge_features = half_edge_features.view(num_batches * num_half_edges_new, num_channels)
        return half_edge_features
