import cv2
from src.detect_danger import detect_danger
from src.preprocess import preprocess
from src.detect_enemies import detect_enemies
from src.detect_player import detect_player
import numpy as np
from src.dash import get_dash_coords, DASH_DISTANCE
from src.action import get_action
from src.track import track_objects


prev_player = None
prev_small_objects = None


def get_policy(
    frame,
    draw_enemies=False,
    crop_black=False,
    debug_dash=False,
    debug_move=False,
    draw_player=True,
    debug_danger=False,
    debug_enemies=False,
    debug_tracks=False,
    use_tracking=True,
    debug_tracking_points=False,
    debug_simple_enemies=False,
    debug_enemy_contours=False,
    draw_enemy_rects=False,
    draw_enemy_side_points=False,
) -> tuple[list[list[int, int, int]], list[int | None, int | None], bool]:
    global prev_player, prev_small_objects
    frame = preprocess(frame, crop_black=crop_black)
    h, w, _ = frame.shape
    center = (w // 2, h // 2)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    player = detect_player(hsv_frame)
    if player is None:
        player = prev_player
    if player is None:
        return frame, (None, None), False
    prev_player = player
    xp, yp = player
    enemies, radiuses, rects, rect_radiuses, enemy_contours, small_objects = detect_enemies(
        frame, hsv_frame, player, draw_rects=draw_enemy_rects, draw_side_points=draw_enemy_side_points
    )
    dangerous_countours, dangerous_rects, dangerous_radiuses = detect_danger(
        hsv_frame, player
    )

    all_enemies = enemies + rects + dangerous_rects
    all_radiuses = radiuses + rect_radiuses + dangerous_radiuses

    if use_tracking:
        small_objects = np.array(small_objects)
        tracked_points, tracked_radiuses = track_objects(frame, small_objects, prev_small_objects, player, debug_lines=debug_tracks)
        prev_small_objects = small_objects
        all_enemies.extend(tracked_points)
        all_radiuses.extend(tracked_radiuses)
        if debug_tracking_points:
            for point, radius in zip(tracked_points, tracked_radiuses):
                cv2.circle(frame, (point[0], point[1]), int(radius) if radius > 0 else 5, (0, 0, 255), -1)

    all_enemies = np.array(all_enemies)
    all_radiuses = np.array(all_radiuses)

    if debug_enemy_contours:
        cv2.drawContours(frame, enemy_contours, -1, (0, 255, 255), 2)

    if debug_simple_enemies:
        for enemy, radius in zip(enemies, radiuses):
            cv2.circle(
                frame,
                (enemy[0], enemy[1]),
                int(radius) if radius > 0 else 5,
                (0, 0, 255),
                -1,
            )

    if draw_enemies:
        for enemy, radius in zip(all_enemies, all_radiuses):
            cv2.circle(
                frame,
                (enemy[0], enemy[1]),
                int(radius) if radius > 0 else 5,
                (0, 255, 0),
                -1,
            )

    if draw_player:
        cv2.circle(frame, (xp, yp), 10, (0, 255, 255), -1)

    if debug_danger:
        cv2.drawContours(frame, dangerous_countours, -1, (0, 255, 255), 2)
    if debug_enemies:
        cv2.drawContours(frame, enemy_contours, -1, (255, 0, 255), 2)

    is_dash = False
    new_x, new_y = None, None
    if len(enemies) > 0:
        dash_coords = get_dash_coords(
            frame,
            player,
            dangerous_countours,
            enemy_contours,
            all_enemies,
            all_radiuses,
            debug=debug_dash,
        )
        if dash_coords is not None:
            new_x, new_y = dash_coords
            is_dash = True
            if debug_move:
                cv2.circle(
                    frame,
                    (xp + new_x * DASH_DISTANCE, yp + new_y * DASH_DISTANCE),
                    10,
                    (255, 255, 0),
                    -1,
                )
        else:
            new_x, new_y = get_action(player, all_enemies, all_radiuses, center)
            if debug_move:
                cv2.circle(
                    frame, (xp + new_x * 10, yp + new_y * 10), 10, (255, 255, 0), -1
                )
    return frame, (new_x, new_y), is_dash
