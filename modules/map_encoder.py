import torch
import torch.nn as nn

from utils.train_utils import weight_init
from utils.layers import MLPLayer

class MapEncoder(nn.Module):
    def __init__(self, cfg):
        super(MapEncoder, self).__init__()
        self.cfg = cfg 
        self.cfg_model = cfg.model 
        self.cfg_rl_waymo = cfg.dataset.waymo

        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.map_seeds = nn.Parameter(torch.Tensor(1, 1, self.cfg_model.hidden_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.map_seeds)
        self.road_pts_encoder = MLPLayer(self.cfg_model.map_attr, self.cfg_model.hidden_dim, self.cfg_model.hidden_dim)
        self.road_pts_attn_layer = nn.MultiheadAttention(self.cfg_model.hidden_dim, num_heads=self.cfg_model.num_heads, dropout=self.cfg_model.dropout)
        self.norm1 = nn.LayerNorm(self.cfg_model.hidden_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(self.cfg_model.hidden_dim, eps=1e-5)
        self.map_feats = MLPLayer(self.cfg_model.hidden_dim, self.cfg_model.hidden_dim, self.cfg_model.hidden_dim)
        self.road_type_encoder = MLPLayer(self.cfg_model.num_road_types, self.cfg_model.hidden_dim, self.cfg_model.hidden_dim)
        self.road_road_type_encoder = MLPLayer(self.cfg_model.hidden_dim * 2, self.cfg_model.hidden_dim, self.cfg_model.hidden_dim)

        self.apply(weight_init)

    def get_road_pts_mask(self, roads):
        road_segment_mask = torch.sum(roads[:, :, :, -1], dim=2) == 0
        road_pts_mask = (1.0 - roads[:, :, :, -1]).type(torch.BoolTensor).to(roads.device).view(-1, roads.shape[2])
        road_pts_mask[:, 0][road_pts_mask.sum(-1) == roads.shape[2]] = False  # Ensures no NaNs due to empty rows.
        return road_segment_mask, road_pts_mask

    def forward(self, data):
        road_points = data['map'].road_points.float()
        road_types = data['map'].road_types.float()

        batch_size = road_points.shape[0]
        # [batch_size, num_polylines], [batch_size * num_polylines, num_points_per_polyline]
        road_segment_mask, road_pts_mask = self.get_road_pts_mask(road_points)
        road_pts_feats = self.road_pts_encoder(road_points[:, :, :, :self.cfg_model.map_attr]).view(batch_size*self.cfg_rl_waymo.max_num_road_polylines, self.cfg_rl_waymo.max_num_road_pts_per_polyline, -1).permute(1, 0, 2)
        road_type_feats = self.road_type_encoder(road_types).unsqueeze(0).reshape(1, batch_size*self.cfg_rl_waymo.max_num_road_polylines, -1)
        map_seeds = self.map_seeds.repeat(1, batch_size*self.cfg_rl_waymo.max_num_road_polylines, 1)
        road_seg_emb = self.road_pts_attn_layer(query=map_seeds, key=road_pts_feats, value=road_pts_feats,
                                                key_padding_mask=road_pts_mask)[0]
        road_seg_emb = self.norm1(road_seg_emb)
        road_seg_emb2 = road_seg_emb + self.map_feats(road_seg_emb)
        road_seg_emb2 = self.norm2(road_seg_emb2)
        road_seg_emb2 = torch.cat((road_seg_emb2, road_type_feats), dim=-1)
        road_seg_emb2 = self.road_road_type_encoder(road_seg_emb2)
        road_seg_emb = road_seg_emb2.view(1, batch_size, self.cfg_rl_waymo.max_num_road_polylines, -1)[0]
        road_segment_mask = ~road_segment_mask 

        return road_seg_emb, road_segment_mask