import transformers
from transformers.image_transforms import np, Union, Iterable, Optional, ChannelDimension, \
    infer_channel_dimension_format, get_channel_dimension_axis, to_channel_dimension_format

def patched_normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
    data_format: Optional[ChannelDimension] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:
    """
    Normalizes `image` using the mean and standard deviation specified by `mean` and `std`.

    image = (image - mean) / std

    Args:
        image (`np.ndarray`):
            The image to normalize.
        mean (`float` or `Iterable[float]`):
            The mean to use for normalization.
        std (`float` or `Iterable[float]`):
            The standard deviation to use for normalization.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the output image. If unset, will use the inferred format from the input.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")

    input_data_format = infer_channel_dimension_format(image)
    channel_axis = get_channel_dimension_axis(image)
    num_channels = image.shape[channel_axis]

    if isinstance(mean, Iterable):
        if len(mean) != num_channels:
            if num_channels == 1:
                num_channels = 3
                image = np.concatenate([image, image, image], axis=channel_axis)
            else:
                raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {len(mean)}")
    else:
        mean = [mean] * num_channels
    mean = np.array(mean, dtype=image.dtype)

    if isinstance(std, Iterable):
        if len(std) != num_channels:
            raise ValueError(f"std must have {num_channels} elements if it is an iterable, got {len(std)}")
    else:
        std = [std] * num_channels
    std = np.array(std, dtype=image.dtype)

    if input_data_format == ChannelDimension.LAST:
        image = (image - mean) / std
    else:
        image = ((image.T - mean) / std).T

    image = to_channel_dimension_format(image, data_format) if data_format is not None else image
    return image


def patch_normalize_preprocess():
    transformers.image_transforms.normalize = patched_normalize