import os
import cv2


def extract_target_frames(video_path, output_dir, target_count=150):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео -> {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"Ошибка: видео пустое или повреждено -> {video_path}")
        cap.release()
        return

    frame_skip = max(1, total_frames // target_count)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_filename = os.path.join(output_dir, f"{video_name}_f{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"[Успех] Из видео '{video_name}' сохранено кадров: {saved_count} в папку '{output_dir}'")


datasets = {
    "sunset": [
        "sun3.mp4",
    ],
    "modern": [
        "mod3.mp4",
    ],
    "indoor": [
        "indoor1.mp4",
    ],
    "night": [
        "ab.mp4",
    ]
}

for category, video_paths in datasets.items():
    output_folder = f"dataset_output/{category}"
    print(f"\n--- Обработка категории: {category.upper()} ---")

    for video_path in video_paths:
        if os.path.exists(video_path):
            extract_target_frames(video_path, output_folder, target_count=180)
        else:
            print(f"[Пропущено] Файл не найден: {video_path}")

print("\nНарезка всех датасетов завершена!")