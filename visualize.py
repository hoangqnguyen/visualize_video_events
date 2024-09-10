import os
import cv2
import glob
import random
import numpy as np
from utils import (
    load_json,
    filter_events_by_score,
    non_max_suppression_events,
)


def draw_filled_parallelogram(text, center_coordinates, image, dt):
    # Initial parameters
    radius = 10
    color = (255, 255, 255)  # White color in BGR
    circle_thickness = 1  # Solid circle
    delta_shadow = np.array([3, 3])
    color_shadow = (57, 61, 71)  # Black color for the shadow
    color_text = (0, 0, 0)  # Black color for the text

    # Text and padding
    font_scale = 0.7
    font_thickness = 1
    # font = cv2.FONT_HERSHEY_SIMPLEX
    font = cv2.FONT_HERSHEY_COMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    left_padding = 7
    right_padding = 20
    top_padding = 12
    bottom_padding = 7

    # Adjust the length of the parallelogram based on the text size and padding
    parallelogram_length = text_size[0] + left_padding + right_padding
    parallelogram_height = text_size[1] + top_padding + bottom_padding

    # Define angle for the 60-degree line
    angle = np.pi / 3  # 60 degrees in radians
    positive_angle = 2 * np.pi / 3  # 120 degrees in radians

    # Calculate the 60-degree line points
    start_point = center_coordinates
    end_point = (
        int(center_coordinates[0] + 100 * np.cos(angle)),
        int(center_coordinates[1] - 100 * np.sin(angle)),
    )

    # Calculate the points of the parallelogram, with the first point being below the 60-degree line
    gap = 5
    point1 = (end_point[0] + gap, end_point[1] + gap)  # Below the 60-degree line
    point2 = (
        int(point1[0] + parallelogram_length),
        point1[1],
    )  # Move horizontally (parallel to horizontal line)
    point3 = (
        int(point2[0] + parallelogram_height * np.cos(positive_angle)),
        int(point2[1] + parallelogram_height * np.sin(positive_angle)),
    )  # Adjusted to 120 degrees
    point4 = (
        int(point1[0] + parallelogram_height * np.cos(positive_angle)),
        int(point1[1] + parallelogram_height * np.sin(positive_angle)),
    )  # Adjusted to 120 degrees

    # Draw the white dot with radius 10
    cv2.circle(
        image,
        center_coordinates,
        radius,
        color,
        circle_thickness,
        lineType=cv2.LINE_AA,
    )

    # Draw the 60-degree line
    end_point_ = np.add(
        start_point, np.subtract(end_point, center_coordinates) * min(1.0, dt * 4)
    ).astype(np.int32)

    start_point_shadow = np.add(start_point, delta_shadow * 0.5).astype(np.int32)
    end_point_shadow = np.add(end_point_, delta_shadow * 0.5).astype(np.int32)

    cv2.line(
        image,
        start_point_shadow,
        end_point_shadow,
        color_shadow,  # Black color for the shadow
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    cv2.line(image, start_point, end_point_, color, thickness=2, lineType=cv2.LINE_AA)

    # Draw the horizontal line from the end of the first line

    if dt >= 1 / 4.0:
        horizontal_end_point = (
            int(end_point[0] + 60 * min(1.0, 4 * (dt - 0.25))),
            end_point[1],
        )
        horizontal_end_point_shadow = np.add(
            horizontal_end_point, delta_shadow * 0.5
        ).astype(np.int32)
        cv2.line(
            image,
            end_point_shadow,
            horizontal_end_point_shadow,
            color_shadow,  # Black color for the shadow
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cv2.line(
            image,
            end_point,
            horizontal_end_point,
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    # Fill the parallelogram by using the fillPoly function
    delta = np.array([max(0, 1 - dt * 4) * 15, 0])
    pts = np.array([point1, point2, point3, point4], np.int32)
    pts = (pts + delta).astype(np.int32)
    pts_shadow = (pts + delta_shadow).astype(np.int32)
    pts = pts.reshape((-1, 1, 2))
    pts_shadow = pts_shadow.reshape((-1, 1, 2))

    # draw main parallelogram
    cv2.fillPoly(image, [pts_shadow], color_shadow, lineType=cv2.LINE_AA)
    cv2.fillPoly(image, [pts], color, lineType=cv2.LINE_AA)

    # Calculate text position (centered inside the parallelogram with adjusted padding)
    text_x = pts[0][0][0] + left_padding
    text_y = pts[0][0][1] + top_padding + text_size[1] // 2

    # Add the text inside the parallelogram
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        font_scale,
        color_text,
        font_thickness,
        lineType=cv2.LINE_AA,
    )

    return image


def generate_video(config):
    num_videos = config["num_videos"]
    preds = config["preds"]
    output_dir = config["output_dir"]
    frame_dir = config["frame_dir"]
    fps = config["fps"]
    width = config["width"]
    height = config["height"]
    freeze_frames = config["freeze_frames"]
    fourcc = config["fourcc"]
    main_color = config["main_color"]

    for _ in range(num_videos):
        chosen_video = random.choice(preds)
        print(f"Chosen video: {chosen_video['video']}")
        event_by_frame = {x["frame"]: x for x in chosen_video["events"]}
        output_video = os.path.join(output_dir, chosen_video["video"] + ".mp4")

        org_frames_path = sorted(
            glob.glob(os.path.join(frame_dir, chosen_video["video"], "*.jpg"))
        )

        frames_im = []
        for i, frame_path in enumerate(org_frames_path):
            img = cv2.imread(frame_path)
            # resize if needed
            if img.shape[1] != width or img.shape[0] != height:
                img = cv2.resize(img, (width, height))

            # check if event in frame
            if i not in event_by_frame:
                frames_im.append(img)
            else:
                event = event_by_frame[i]
                x, y = int(event["xy"][0] * width), int(event["xy"][1] * height)
                score = event["score"]
                label = event["label"]
                text = f"{label.upper()} {score:.0%}"

                # freeze frames to animate annotations
                for it in range(freeze_frames):
                    dt = it / freeze_frames
                    canvas = img.copy()  # copy frame

                    draw_filled_parallelogram(text, (x, y), canvas, dt)

                    alpha = max(0.9, 0.65 + 0.25 * dt)

                    # blend frame with annotation
                    canvas = cv2.addWeighted(img, 1 - alpha, canvas, alpha, 0)

                    # draw circle after blending to avoid blending issues
                    radius = 10 + 40 * (1 - dt) ** 8  # radius decreases with time
                    thickness = 1 + dt * 2  # thickness increases with time
                    cv2.circle(
                        canvas,
                        (x, y),
                        int(radius),
                        main_color,
                        int(thickness),
                        lineType=cv2.LINE_AA,
                    )

                    frames_im.append(canvas)

        # write to video
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        for _, frame_path in enumerate(frames_im):
            video_writer.write(frame_path)

        video_writer.release()
        print("Video saved to", output_video)


if __name__ == "__main__":
    preds = load_json("exp/pred-val.138.recall.json")
    
    # filter and nms preds
    preds = filter_events_by_score(preds, 0.5)
    preds = non_max_suppression_events(preds, 3)
    # Example call
    config = {
        "num_videos": 20,
        "preds": preds,
        "output_dir": "exp/demo_best",
        "frame_dir": "data/kovo_hd/frames",
        "fps": 24,
        "width": 1280,
        "height": 720,
        "freeze_frames": 24 // 1,
        "fourcc": cv2.VideoWriter_fourcc(*"mp4v"),
        "main_color": (242, 243, 244),
    }
    os.makedirs(config["output_dir"], exist_ok=True)

    generate_video(config)
