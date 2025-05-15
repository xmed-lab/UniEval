import cv2
import glob
import numpy as np
import os
import requests
import PIL
import PIL.Image

from collections import defaultdict
from transformers import PretrainedConfig
from typing import Any, Dict, List, Optional, Union

from ..constants import DEFAULT_IMAGE_TOKEN
from ..media import Image, Video
from ..utils import make_list
from ..utils.logging import logger

__all__ = ["extract_media"]


def _extract_image(image: Union[Image, PIL.Image.Image]) -> PIL.Image.Image:
    if isinstance(image, Image):
        if image.path.startswith("http://") or image.path.startswith("https://"):
            image = PIL.Image.open(requests.get(image.path, stream=True).raw)
        else:
            image = PIL.Image.open(image.path)
    return image


def _load_video(video_path: str, *, num_frames: int) -> List[PIL.Image.Image]:
    # Load video frames from a directory
    if os.path.isdir(video_path):
        frame_paths = sorted(glob.glob(os.path.join(video_path, "*")))
        indices = np.round(np.linspace(0, len(frame_paths) - 1, num_frames)).astype(int)
        return [PIL.Image.open(frame_paths[index]) for index in indices]

    # Load video frames from a video file
    vidcap = cv2.VideoCapture(video_path)

    # Find the last frame as frame count might not be accurate
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_count > 0:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        if vidcap.grab():
            break
        frame_count -= 1
    else:
        raise ValueError(f"Video '{video_path}' has no frames.")
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Extract frames uniformly
    indices = np.round(np.linspace(0, frame_count - 1, num_frames)).astype(int)
    frames = {}
    for index in range(frame_count):
        success = vidcap.grab()
        if not success:
            raise ValueError(f"Failed to grab frame {index} from video '{video_path}'.")
        if index not in indices:
            continue
        success, frame = vidcap.retrieve()
        if not success:
            logger.warning(f"Failed to retrieve frame {index} from video '{video_path}'. Skipped.")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[index] = PIL.Image.fromarray(frame)
    return [frames[index] for index in indices if index in frames]


def _extract_video(video: Video, config: PretrainedConfig) -> List[PIL.Image.Image]:
    num_frames = config.num_video_frames

    frames = _load_video(video.path, num_frames=num_frames)
    return frames


def extract_media(
    messages: List[Dict[str, Any]],
    config: Optional[PretrainedConfig] = None,
    draft: bool = False,
) -> Dict[str, List[Any]]:
    media = defaultdict(list)
    for message in messages:
        text = ""
        for part in make_list(message["value"]):
            if isinstance(part, str):
                text += part
            elif isinstance(part, (Image, PIL.Image.Image)):
                if draft:
                    media["image"].append(part)
                else:
                    image = _extract_image(part)
                    text += DEFAULT_IMAGE_TOKEN + "\n"
                    media["image"].append(image)
            elif isinstance(part, Video):
                if draft:
                    media["video"].append(part)
                else:
                    video = _extract_video(part, config)
                    text += (DEFAULT_IMAGE_TOKEN + "\n") * len(video)
                    media["image"].extend(video)
            else:
                raise ValueError(f"Unsupported prompt part type: {type(part)}")
        message["value"] = text

    return media