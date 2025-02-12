import cv2
import math

def createReferenceLine(frames):
    output_frames = []
    for frame in frames:
        # Draw reference line
        cv2.line(frame, (820, 720), (820, 0), (0, 0, 0), 2)
        output_frames.append(frame)
    return output_frames

# def detect_ball_speed(ball_positions, reference_line=820, fps=30):
#     """
#     計算球越過參考線後的移動速度

#     參數:
#       ball_positions - 每幀球的中心座標列表，格式為 [(x, y), (x, y), ...]
#       reference_line - 參考線的 x 座標
#       fps - 每秒幀數

#     回傳:
#       speeds - 列表形式，每個元素為 (cross_frame, next_frame, speed_in_pixels_per_second)
#     """
#     speeds = []
#     crossing_index = None
#     crossing_position = None

#     # 遍歷所有幀，檢查球是否從參考線一側越過到另一側
#     for i in range(1, len(ball_positions)):
#         prev = ball_positions[i - 1]
#         curr = ball_positions[i]
#         # 檢查如果球之前在參考線左側且目前在右側，或反之亦然
#         if crossing_index is None and ((prev[0] < reference_line and curr[0] >= reference_line) or (prev[0] > reference_line and curr[0] <= reference_line)):
#             crossing_index = i
#             crossing_position = curr
#         # 當已記錄越線後，等待下一個球位置來計算速度
#         elif crossing_index is not None:
#             frame_diff = i - crossing_index
#             # 計算兩點之間的歐氏距離
#             dx = curr[0] - crossing_position[0]
#             dy = curr[1] - crossing_position[1]
#             distance = math.sqrt(dx**2 + dy**2)
#             # 計算經過時間，單位為秒，進而計算速度 (像素/秒)
#             time_elapsed = frame_diff / fps
#             if time_elapsed > 0:
#                 speed = distance / time_elapsed
#                 speeds.append((crossing_index, i, speed))
#             # 重設交界事件
#             crossing_index = None
#             crossing_position = None

#     return speeds

def detect_ball_speed(video_frames, player_detections, reference_Line=820, fps=30):
    output_video_frames = []
    # 用來記錄球在參考線左側的最後一個 x2 座標，資料結構：{track_id: x2}
    prev_positions = {}

    for frame, ball_dict in zip(video_frames, player_detections):
        # 遍歷每個偵測到的球
        for track_id, bbox in ball_dict.items():
            x1, y1, x2, y2 = map(int, bbox)
            
            # 當球位於參考線左側時，紀錄目前的 x2 座標
            if x2 < reference_Line:
                prev_positions[track_id] = x2
            
            # 當球位於參考線右側時，若之前有紀錄則計算像素距離
            elif x1 > reference_Line:
                if track_id in prev_positions:
                    speed = x1 - prev_positions[track_id]
                    # 在畫面上顯示速度資訊
                    speed = x1 - prev_positions[track_id]
                    # 準備文字內容
                    text = f"Speed: {speed:.2f} pixels/sec"
                    # 取得文字尺寸 (字寬, 字高)
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    # 設定右上角擺放位置，保留 10 像素邊距
                    x_pos = frame.shape[1] - text_width - 10
                    y_pos = frame.shape[0] - 10
                    # 畫一個黑色背景方塊，讓文字更清晰
                    cv2.rectangle(frame, (x_pos - 5, y_pos - text_height - 5), (x_pos + text_width + 5, y_pos + 5), (0, 0, 0), cv2.FILLED)
                    # 在背景上寫入白色文字
                    cv2.putText(frame, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                else:
                    # 若找不到 prev_positions，則不做計算
                    speed = x1 - prev_positions[track_id]
                    # 在畫面上顯示速度資訊
                    speed = x1 - prev_positions[track_id]
                    # 準備文字內容
                    text = f"Speed: {speed:.2f} pixels/sec"
                    # 取得文字尺寸 (字寬, 字高)
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    # 設定右上角擺放位置，保留 10 像素邊距
                    x_pos = frame.shape[1] - text_width - 10
                    y_pos = frame.shape[0] - 10
                    # 畫一個黑色背景方塊，讓文字更清晰
                    cv2.rectangle(frame, (x_pos - 5, y_pos - text_height - 5), (x_pos + text_width + 5, y_pos + 5), (0, 0, 0), cv2.FILLED)
                    # 在背景上寫入白色文字
                    cv2.putText(frame, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            else:
                # 如果球在參考線正附近，不進行任何處理
                pass

        output_video_frames.append(frame)

    return output_video_frames