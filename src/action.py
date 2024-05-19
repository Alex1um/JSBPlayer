import numpy as np


def get_action(player_pos: tuple[int, int], enemy_poses: list[tuple[int, int]], enemy_rects: list[tuple[int, int, int, int]], radiuses: list[int], center: tuple[int, int], search_radius=150) -> tuple[int, int] | None:

    radiuses = np.array(radiuses)
    player_pos = np.array(player_pos)
    enemy_poses = np.array(enemy_poses)

    lengths = np.linalg.norm(enemy_poses - player_pos, axis=1) - radiuses
    indexes = lengths < search_radius
    if not np.any(indexes):
        return np.sign(center[0] - player_pos[0]), np.sign(center[1] - player_pos[1])
    lengths = lengths[indexes]
    enemy_poses = enemy_poses[indexes]
    
    weighed_avg_dx = -np.average(enemy_poses[:, 0] - player_pos[0], weights=100/(lengths**2))
    weighed_avg_dy = -np.average(enemy_poses[:, 1] - player_pos[1], weights=100/(lengths**2))
    weighed_avg_dx += (center[0] - player_pos[0]) / 100
    weighed_avg_dy += (center[1] - player_pos[0]) / 100
    # print(weighed_avg_dx, weighed_avg_dy)
    result_angle = np.arctan2(weighed_avg_dy, weighed_avg_dx)

    # min_length = np.argmin(lengths)
    # min_angle = np.arctan2(player_pos[1] - enemy_poses[min_length, 1], player_pos[0] - enemy_poses[min_length, 0])
    # result_angle = min_angle + 2 * np.pi if min_angle < 0 else min_angle
    val_x = np.sign(np.cos(result_angle))
    val_y = np.sign(np.sin(result_angle))
    if np.nan == val_x:
        val_x = 0
    if np.nan == val_y:
        val_y = 0
    action_x = int(val_x)
    action_y = int(val_y)

    # x = round(player_pos[0] + count * np.cos(result_angle))
    # y = round(player_pos[1] + count * np.sin(result_angle))

    return action_x, action_y

