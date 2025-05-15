from openai import OpenAI  # OpenAI Python library to make API calls
import openai
import requests  # used to download images
import os  # used to access filepaths
from PIL import Image  # used to print and edit images


class DALLE2:

    def __init__(self, model_name='dall-e-2'):
        API_SECRET_KEY = '[Your Key]'
        BASE_URL = '[Base URL]'
        self.client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
        self.model_name = model_name

    def generate(self, input_text, img_num, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        outputs = []
        for i in range(img_num):
            generation_response = 'No Response!'
            try:
                generation_response = self.client.images.generate(
                    model = self.model_name,
                    prompt=input_text,
                    n=1,
                    size="1024x1024",
                    response_format="url",
                )

                open(os.path.join(save_dir, f"response_{i}.txt"), 'w').write(str(generation_response))
                path = os.path.join(save_dir, f"image_{i}.png")
                generated_image_url = generation_response.data[0].url
                generated_image = requests.get(generated_image_url).content
                with open(path, "wb") as image_file:
                    image_file.write(generated_image)
                outputs.append(path)

            except:
                print(generation_response)
                outputs.append('')

        return outputs


class DALLE3:

    def __init__(self, model_name='dall-e-3'):
        API_SECRET_KEY = '[Your Key]'
        BASE_URL = '[Base URL]'
        self.client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
        self.model_name = model_name

    def generate(self, input_text, img_num, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        outputs = []
        for i in range(img_num):
            generation_response = 'No Response!'
            try:
                generation_response = self.client.images.generate(
                    model = self.model_name,
                    prompt=input_text,
                    n=1,
                    size="1024x1024",
                    response_format="url",
                )

                open(os.path.join(save_dir, f"response_{i}.txt"), 'w').write(str(generation_response))
                path = os.path.join(save_dir, f"image_{i}.png")
                generated_image_url = generation_response.data[0].url
                generated_image = requests.get(generated_image_url).content
                with open(path, "wb") as image_file:
                    image_file.write(generated_image)
                outputs.append(path)

            except:
                print(generation_response)
                outputs.append('')

        return outputs
