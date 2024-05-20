import cv2
import numpy as np

tracker = None


def closest_point_on_ray(point, ray_point1, ray_point2):
    # Calculate the direction vector of the ray
    direction = np.subtract(ray_point2, ray_point1)

    # Calculate the vector from the first point of the ray to the given point
    vector_to_point = np.subtract(point, ray_point1)

    # Calculate the dot product of the direction vector and the vector from the first point to the given point
    dot_product = np.dot(direction, vector_to_point)

    # If the dot product is negative, the given point is behind the ray, so the closest point is the first point of the ray
    if dot_product < 0:
        return None
    else:
        # Calculate the length squared of the direction vector
        length_squared = np.dot(direction, direction)

        # Divide the dot product by the length squared to get the parameter t
        t = dot_product / length_squared

        # Calculate the closest point on the ray
        closest_point = np.add(ray_point1, np.multiply(t, direction))

    return closest_point.astype(int)


def track_objects(
    frame,
    small_objects: np.ndarray[int, int],
    prev_small_objects: np.ndarray[int, int],
    player,
    debug_lines=False,
):
    if (
        prev_small_objects is None
        or len(small_objects) == 0
        or len(prev_small_objects) == 0
    ):
        return [], []
    h, w, *_ = frame.shape
    enemies = []
    ranges = []
    player = np.array(player)
    for object in small_objects:
        min_dist = np.argmin(np.linalg.norm(prev_small_objects - object, axis=1))
        prev_object = prev_small_objects[min_dist]
        if debug_lines:
            cv2.arrowedLine(
                frame,
                (prev_object[0], prev_object[1]),
                (object[0], object[1]),
                (255, 255, 255),
                2,
            )
        on_line = closest_point_on_ray(player, prev_object, object)
        if (
            on_line is None
            or on_line[0] < 0
            or on_line[0] >= w
            or on_line[1] < 0
            or on_line[1] >= h
        ):
            continue
        enemies.append(on_line)
        ranges.append(-10)
    return enemies, ranges
