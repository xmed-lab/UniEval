import argparse
import cv2
import numpy as np
import os
import vila_u


def save_image(response, path):
    os.makedirs(path, exist_ok=True)
    for i in range(response.shape[0]):
        image = response[i].permute(1, 2, 0)
        image = image.cpu().numpy().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path, f"image_{i}.png"), image)


def save_video(response, path):
    os.makedirs(path, exist_ok=True)
    for i in range(response.shape[0]):
        video = response[i].permute(0, 2, 3, 1)
        video = video.cpu().numpy().astype(np.uint8)
        video = np.concatenate(video, axis=1)
        video = cv2.cvtColor(video, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path, f"video_{i}.png"), video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    ### image/video understanding arguments
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.9, help="The value of temperature for text generation.")
    parser.add_argument("--top_p", type=float, default=0.6, help="The value of top-p for text generation.")
    ### image and video generation arguments
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--video_generation", type=bool, default=False)
    parser.add_argument("--cfg", type=float, default=3.0, help="The value of the classifier free guidance for image generation.")
    parser.add_argument("--save_path", type=str, default="generated_images/")
    parser.add_argument("--generation_nums", type=int, default=1)
    args = parser.parse_args()

    if args.model_path is not None:
        model = vila_u.load(args.model_path)
    else:
        raise ValueError("No model path provided!")

    if args.query is not None:
        generation_config = model.default_generation_config
        generation_config.temperature = args.temperature
        generation_config.top_p = args.top_p
        if args.image_path is not None:
            image = vila_u.Image(args.image_path)
            response = model.generate_content([image, args.query])
            print("\033[1;32mResponse:\033[0m", response)
            exit()
        elif args.video_path is not None:
            video = vila_u.Video(args.video_path)
            response = model.generate_content([video, args.query])
            print("\033[1;32mResponse:\033[0m", response)
            exit()
        else:
            raise ValueError("No visual content input!")
    elif args.prompt is not None:
        if args.video_generation:
            response = model.generate_video_content(args.prompt, args.cfg, args.generation_nums)
            save_video(response, args.save_path)
            exit()
        else:
            response = model.generate_image_content(args.prompt, args.cfg, args.generation_nums)
            save_image(response, args.save_path)
            exit()
    else:
        raise ValueError("No query or prompt provided!")