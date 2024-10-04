import numpy as np
import torch
np.set_printoptions(suppress=True)
from utils.geometry import dot_product_2d, cross_product_2d
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import BaseStorage, EdgeStorage, NodeStorage

# Constant distance to apply when distances are invalid. This will avoid the
# propagation of nans and should be reduced out when taking the maximum anyway.
_EXTREMELY_LARGE_DISTANCE = 1e10
# Off-road threshold, i.e. smallest distance away from the road edge that is
# considered to be a off-road.
OFFROAD_DISTANCE_THRESHOLD = 0.0

# How close the start and end point of a map feature need to be for the feature
# to be considered cyclic, in m^2.
_CYCLIC_MAP_FEATURE_TOLERANCE_M2 = 1.0

def compute_distance_to_road_edge_gpu(center_x, center_y, road_edge_polylines):
    # Concatenate tensors to have the same convention as `box_utils`.
    boxes = torch.stack([center_x, center_y], dim=-1)
    num_objects, num_steps, num_features = boxes.shape

    # Flatten query points.
    # `flat_eval_corners` shape: (num_query_points, 2).
    flat_eval_corners = boxes.reshape(-1, 2)

    # Tensorize road edges.
    polyline_tensors = []
    for polyline in road_edge_polylines:
        polyline_tensors.append([[map_point[0], map_point[1]] for map_point in polyline])

    # Compute distances for all query points.
    # `corner_distance_to_road_edge` shape: (num_query_points).
    corner_distance_to_road_edge = _compute_signed_distance_to_polylines_gpu(
        xys=flat_eval_corners, polylines=polyline_tensors
    )

    return corner_distance_to_road_edge


def _compute_signed_distance_to_polylines_gpu(xys, polylines):
    """Computes the signed distance to the 2D boundary defined by polylines.

    Negative distances correspond to being inside the boundary (e.g. on the
    road), positive distances to being outside (e.g. off-road).

    The polylines should be oriented such that port side is inside the boundary
    and starboard is outside, a.k.a counterclockwise winding order.

    Note: degenerate segments (start == end) can cause undefined behaviour.

    Args:
      xys: A float Tensor of shape (num_points, 2) containing xy coordinates of
        query points.
      polylines: List of tensors of shape (num_segments+1, 2) containing sequences
        of xy coordinates representing start and end points of consecutive
        segments.

    Returns:
      A tensor of shape (num_points), containing the signed distance from queried
        points to the nearest polyline.
    """
    distances = []
    for polyline in polylines:
        # Skip degenerate polylines.
        if len(polyline) < 2:
            continue
        polyline = torch.tensor(polyline).cuda()
        distances.append(_compute_signed_distance_to_polyline_gpu(xys, polyline))

    # `distances` shape: (num_points, num_nondegenerate_polylines).
    distances = torch.stack(distances, axis=-1)
    return torch.gather(distances, 1, torch.argmin(torch.abs(distances), dim=-1)[:, None])[:, 0]
    # return np.take_along_axis(distances, np.argmin(np.abs(distances), axis=-1)[:, None], axis=1)[:, 0]


def _compute_signed_distance_to_polyline_gpu(xys, polyline):
    """Computes the signed distance to the 2D boundary defined by a polyline.

    Negative distances correspond to being inside the boundary (e.g. on the
    road), positive distances to being outside (e.g. off-road).

    The polyline should be oriented such that port side is inside the boundary
    and starboard is outside, a.k.a counterclockwise winding order.

    Note: degenerate segments (start == end) can cause undefined behaviour.

    Args:
      xys: A float Tensor of shape (num_points, 2) containing xy coordinates of
        query points.
      polyline: A float Tensor of shape (num_segments+1, 2) containing sequences
        of xy coordinates representing start and end points of consecutive
        segments.

    Returns:
      A tensor of shape (num_points), containing the signed distance from queried
        points to the polyline.
    """
    is_cyclic = torch.square(polyline[0] - polyline[-1]).sum() < _CYCLIC_MAP_FEATURE_TOLERANCE_M2
    # Get distance to each segment.
    # shape: (num_points, num_segments, 2)
    xy_starts = polyline[None, :-1, :2]
    xy_ends = polyline[None, 1:, :2]
    start_to_point = xys[:, None, :2] - xy_starts
    start_to_end = xy_ends - xy_starts

    # Relative coordinate of point projection on segment.
    # shape: (num_points, num_segments)
    rel_t = torch.nan_to_num(dot_product_2d(start_to_point, start_to_end) / dot_product_2d(start_to_end, start_to_end))

    # Negative if point is on port side of segment, positive if point on
    # starboard side of segment.
    # shape: (num_points, num_segments)
    n = torch.sign(cross_product_2d(start_to_point, start_to_end))
    # Absolute distance to segment.
    # shape: (n_points, n_segments)
    distance_to_segment = torch.linalg.norm(start_to_point - (start_to_end * torch.clip(rel_t, 0.0, 1.0)[..., None]), axis=-1)

    # There are 3 cases:
    #   - if the point projection on the line falls within the segment, the sign
    #       of the distance is `n`.
    #   - if the point projection on the segment falls before the segment start,
    #       the sign of the distance depends on the convexity of the prior and
    #       nearest segments.
    #   - if the point projection on the segment falls after the segment end, the
    #       sign of the distance depends on the convexity of the nearest and next
    #       segments.

    # shape: (num_points, num_segments+2, 2)
    start_to_end_padded = torch.cat([start_to_end[:, -1:], start_to_end, start_to_end[:, :1]], dim=1)
    # shape: (num_points, num_segments+1)
    is_locally_convex = cross_product_2d(start_to_end_padded[:, :-1], start_to_end_padded[:, 1:]) > 0.0

    # shape: (num_points, num_segments)
    n_prior = torch.cat([torch.where(is_cyclic, n[:, -1:], n[:, :1]), n[:, :-1]], dim=-1)
    n_next = torch.cat([n[:, 1:], torch.where(is_cyclic, n[:, :1], n[:, -1:])], dim=-1)

    # shape: (num_points, num_segments)
    sign_if_before = torch.where(is_locally_convex[:, :-1], torch.maximum(n, n_prior), torch.minimum(n, n_prior))
    sign_if_after = torch.where(is_locally_convex[:, 1:], torch.maximum(n, n_next), torch.minimum(n, n_next))

    # shape: (num_points, num_segments)
    sign_to_segment = torch.where(rel_t < 0.0, sign_if_before, torch.where(rel_t < 1.0, n, sign_if_after))

    # shape: (num_points)
    distance_sign = torch.gather(sign_to_segment, 1, torch.argmin(distance_to_segment, dim=-1)[:, None])[:, 0]
    
    return distance_sign * torch.min(distance_to_segment, dim=-1)[0]


def compute_distance_to_road_edge(center_x, center_y, road_edge_polylines):
    """Computes the distance to the road edge for each of the evaluated objects."""

    # Concatenate tensors to have the same convention as `box_utils`.
    boxes = np.stack([center_x, center_y], axis=-1)
    num_objects, num_steps, num_features = boxes.shape

    # Flatten query points.
    # `flat_eval_corners` shape: (num_query_points, 2).
    flat_eval_corners = np.reshape(boxes, (-1, 2))

    # Tensorize road edges.
    polyline_tensors = []
    for polyline in road_edge_polylines:
        polyline_tensors.append([[map_point[0], map_point[1]] for map_point in polyline])

    # Compute distances for all query points.
    # `corner_distance_to_road_edge` shape: (num_query_points).
    corner_distance_to_road_edge = _compute_signed_distance_to_polylines(
        xys=flat_eval_corners, polylines=polyline_tensors
    )

    # `corner_distance_to_road_edge` shape: (num_evaluated_objects, num_steps, 4).
    # corner_distance_to_road_edge = tf.reshape(
    #     corner_distance_to_road_edge, (1, num_steps, 4)
    # )

    # Reduce to most off-road corner.
    # `signed_distances` shape: (num_evaluated_objects, num_steps).
    # signed_distances = tf.math.reduce_max(corner_distance_to_road_edge, axis=-1)
    return corner_distance_to_road_edge


def _compute_signed_distance_to_polylines(xys, polylines):
    """Computes the signed distance to the 2D boundary defined by polylines.

    Negative distances correspond to being inside the boundary (e.g. on the
    road), positive distances to being outside (e.g. off-road).

    The polylines should be oriented such that port side is inside the boundary
    and starboard is outside, a.k.a counterclockwise winding order.

    Note: degenerate segments (start == end) can cause undefined behaviour.

    Args:
      xys: A float Tensor of shape (num_points, 2) containing xy coordinates of
        query points.
      polylines: List of tensors of shape (num_segments+1, 2) containing sequences
        of xy coordinates representing start and end points of consecutive
        segments.

    Returns:
      A tensor of shape (num_points), containing the signed distance from queried
        points to the nearest polyline.
    """
    distances = []
    for polyline in polylines:
        # Skip degenerate polylines.
        if len(polyline) < 2:
            continue
        polyline = np.array(polyline)
        distances.append(_compute_signed_distance_to_polyline(xys, polyline))

    # `distances` shape: (num_points, num_nondegenerate_polylines).
    distances = np.stack(distances, axis=-1)
    return np.take_along_axis(distances, np.argmin(np.abs(distances), axis=-1)[:, None], axis=1)[:, 0]


def _compute_signed_distance_to_polyline(xys, polyline):
    """Computes the signed distance to the 2D boundary defined by a polyline.

    Negative distances correspond to being inside the boundary (e.g. on the
    road), positive distances to being outside (e.g. off-road).

    The polyline should be oriented such that port side is inside the boundary
    and starboard is outside, a.k.a counterclockwise winding order.

    Note: degenerate segments (start == end) can cause undefined behaviour.

    Args:
      xys: A float Tensor of shape (num_points, 2) containing xy coordinates of
        query points.
      polyline: A float Tensor of shape (num_segments+1, 2) containing sequences
        of xy coordinates representing start and end points of consecutive
        segments.

    Returns:
      A tensor of shape (num_points), containing the signed distance from queried
        points to the polyline.
    """
    is_cyclic = np.square(polyline[0] - polyline[-1]).sum() < _CYCLIC_MAP_FEATURE_TOLERANCE_M2
    # Get distance to each segment.
    # shape: (num_points, num_segments, 2)
    xy_starts = polyline[None, :-1, :2]
    xy_ends = polyline[None, 1:, :2]
    start_to_point = xys[:, None, :2] - xy_starts
    start_to_end = xy_ends - xy_starts

    # Relative coordinate of point projection on segment.
    # shape: (num_points, num_segments)
    rel_t = np.nan_to_num(dot_product_2d(start_to_point, start_to_end) / dot_product_2d(start_to_end, start_to_end))

    # Negative if point is on port side of segment, positive if point on
    # starboard side of segment.
    # shape: (num_points, num_segments)
    n = np.sign(cross_product_2d(start_to_point, start_to_end))
    # Absolute distance to segment.
    # shape: (n_points, n_segments)
    distance_to_segment = np.linalg.norm(start_to_point - (start_to_end * np.clip(rel_t, 0.0, 1.0)[..., None]), axis=-1)

    # There are 3 cases:
    #   - if the point projection on the line falls within the segment, the sign
    #       of the distance is `n`.
    #   - if the point projection on the segment falls before the segment start,
    #       the sign of the distance depends on the convexity of the prior and
    #       nearest segments.
    #   - if the point projection on the segment falls after the segment end, the
    #       sign of the distance depends on the convexity of the nearest and next
    #       segments.

    # shape: (num_points, num_segments+2, 2)
    start_to_end_padded = np.concatenate([start_to_end[:, -1:], start_to_end, start_to_end[:, :1]], axis=1)
    # shape: (num_points, num_segments+1)
    is_locally_convex = cross_product_2d(start_to_end_padded[:, :-1], start_to_end_padded[:, 1:]) > 0.0

    # shape: (num_points, num_segments)
    n_prior = np.concatenate([np.where(is_cyclic, n[:, -1:], n[:, :1]), n[:, :-1]], axis=-1)
    n_next = np.concatenate([n[:, 1:], np.where(is_cyclic, n[:, :1], n[:, -1:])], axis=-1)

    # shape: (num_points, num_segments)
    sign_if_before = np.where(is_locally_convex[:, :-1], np.maximum(n, n_prior), np.minimum(n, n_prior))
    sign_if_after = np.where(is_locally_convex[:, 1:], np.maximum(n, n_next), np.minimum(n, n_next))

    # shape: (num_points, num_segments)
    sign_to_segment = np.where(rel_t < 0.0, sign_if_before, np.where(rel_t < 1.0, n, sign_if_after))

    # shape: (num_points)
    distance_sign = np.take_along_axis(sign_to_segment, np.argmin(distance_to_segment, axis=-1)[:, None], axis=1)[:, 0]
    return distance_sign * np.min(distance_to_segment, axis=-1)

def get_object_type_str(object):
    enum_id = int(object.type)

    if enum_id == 0:
        return "unset"
    elif enum_id == 1:
        return "vehicle"
    elif enum_id == 2:
        return "pedestrian"
    elif enum_id == 3:
        return "cyclist"
    else:
        return "other"

def get_road_type_str(road):
    enum_id = int(road.road_type)

    if enum_id == 0:
        return "none"
    elif enum_id == 1:
        return "lane"
    elif enum_id == 2:
        return "road_line"
    elif enum_id == 3:
        return "road_edge"
    elif enum_id == 4:
        return "stop_sign"
    elif enum_id == 5:
        return "crosswalk"
    elif enum_id == 6:
        return "speed_bump"
    else:
        return "other"

def get_agent_type_onehot(agent_type):
    agent_types = np.eye(3)  # three agent types
    return agent_types[agent_type].tolist()

def get_object_type_onehot(agent_type):
    agent_types = {"unset": 0, "vehicle": 1, "pedestrian": 2, "cyclist": 3, "other": 4}
    return np.eye(len(agent_types))[agent_types[agent_type]]

def get_road_type_onehot(road_type):
    road_types = {"none": 0, "lane": 1, "road_line": 2, "road_edge": 3, "stop_sign": 4, "crosswalk": 5,
                      "speed_bump": 6, "other": 7}
    return np.eye(len(road_types))[road_types[road_type]]

def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data

def add_batch_dim(arr):
    return np.expand_dims(arr, axis=0)

def get_edge_index_complete_graph(graph_size):
    edge_index = torch.cartesian_prod(torch.arange(graph_size, dtype=torch.long),
                                      torch.arange(graph_size, dtype=torch.long)).t()
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]

    return edge_index


def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    else:
        return obj


class MotionData(HeteroData):
    """
    override key `polyline_cluster` indicating which polyline_id is for the vector
    """

    def __inc__(self, key, value, store):
        if 'batch' in key and isinstance(value, Tensor):
            return int(value.max()) + 1
        elif isinstance(store, EdgeStorage) and 'index' in key:
            return torch.tensor(store.size()).view(2, 1)
        elif 'cluster' in key:
            # TODO: remove magic number
            return int(value.min()) + 200
        else:
            return 0
