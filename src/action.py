import numpy as np


def get_action(player_pos: tuple[int, int], enemy_poses: list[tuple[int, int]], count=5) -> tuple[int, int] | None:

    player_pos = np.array(player_pos)
    enemy_poses = np.array(enemy_poses)

    lengths = np.linalg.norm(enemy_poses - player_pos, axis=1)
    
    min_length = np.argmin(lengths)
    min_angle = np.arctan2(player_pos[1] - enemy_poses[min_length, 1], player_pos[0] - enemy_poses[min_length, 0])

    # reversed_angles = np.arctan2(player_pos[1] - enemy_poses[:, 1], player_pos[0] - enemy_poses[:, 0])

    # weighted_result_angle = np.average(reversed_angles, weights=(1/lengths) ** 2)

    result_angle = min_angle
    
    action_x = int(np.sign(np.cos(result_angle)))
    action_y = int(np.sign(np.sin(result_angle)))

    # x = round(player_pos[0] + count * np.cos(result_angle))
    # y = round(player_pos[1] + count * np.sin(result_angle))

    return action_x, action_y

