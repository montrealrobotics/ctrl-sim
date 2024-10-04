import numpy as np

def angle_sub(current_angle, target_angle):
    """Subtract two angles to find the minimum angle between them."""
    # Subtract the angles, constraining the value to [0, 2 * np.pi)
    diff = (target_angle - current_angle) % (2 * np.pi)

    # If we are more than np.pi we're taking the long way around.
    # Let's instead go in the shorter, negative direction
    if diff > np.pi:
        diff = -(2 * np.pi - diff)
    return diff

def angle_sub_tensor(current_angle, target_angle):
    diff = (target_angle - current_angle) % (2 * np.pi)
    mask = diff > np.pi 
    diff[mask] = -(2 * np.pi - diff[mask])

    return diff

def dot_product_2d(a, b):
    """Computes the dot product of 2d vectors."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def cross_product_2d(a, b):
    """Computes the signed magnitude of cross product of 2d vectors."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

def make_2d_rotation_matrix(angle_in_radians):
    """ Makes rotation matrix to rotate point in x-y plane counterclockwise by angle_in_radians.
    """
    return np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                        [np.sin(angle_in_radians), np.cos(angle_in_radians)]])

def apply_se2_transform(coordinates, translation, yaw):
    """
    Converts global coordinates to coordinates in the frame given by the rotation quaternion and
    centered at the translation vector. The rotation is meant to be a z-axis rotation.
    """
    coordinates = coordinates - translation
    
    transform = make_2d_rotation_matrix(angle_in_radians=yaw)
    if len(coordinates.shape) > 2:
        coord_shape = coordinates.shape
        return np.dot(transform, coordinates.reshape((-1, 2)).T).T.reshape(*coord_shape)
    return np.dot(transform, coordinates.T).T[:, :2]

def radians_to_degrees(radians):
    degrees = radians * (180 / 3.141592653589793)
    return degrees