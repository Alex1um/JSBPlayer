import numpy as np


def get_action(player_pos: tuple[int, int], enemy_poses: list[tuple[int, int]], count=5) -> tuple[int, int] | None:

    player_pos = np.array(player_pos)
    enemy_poses = np.array(enemy_poses)

    lengths = np.linalg.norm(enemy_poses - player_pos, axis=1)
    

    reversed_angles = np.arctan2(player_pos[1] - enemy_poses[:, 1], player_pos[0] - enemy_poses[:, 0])
    normalized_angles = np.pi * 2 -  (np.mod(reversed_angles + np.pi, 2 * np.pi) - np.pi)
    weighetd_average = np.average(normalized_angles, weights=(1/lengths) ** 2)
    result_angle = np.mod(weighetd_average + np.pi, 2 * np.pi) - np.pi

    # min_length = np.argmin(lengths)
    # min_angle = np.arctan2(player_pos[1] - enemy_poses[min_length, 1], player_pos[0] - enemy_poses[min_length, 0])
    # result_angle = min_angle + 2 * np.pi if min_angle < 0 else min_angle
    
    action_x = int(np.sign(np.cos(result_angle)))
    action_y = int(np.sign(np.sin(result_angle)))

    # x = round(player_pos[0] + count * np.cos(result_angle))
    # y = round(player_pos[1] + count * np.sin(result_angle))

    return action_x, action_y

