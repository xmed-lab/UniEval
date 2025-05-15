import argparse
import cv2
import gradio as gr
import imghdr
import numpy as np
import os
import shutil
import signal
import sys
import torch
import uuid
import vila_u

CFG = 3.0
TEMPERATURE = 0.9
TOP_P = 0.6


def is_image_file(filepath):
    return imghdr.what(filepath) is not None


def generate_response(image, video, query, chat_history):
    if query is not None and image is None and video is None:
        response = model.generate_image_content(prompt=query, cfg=CFG)[0]
        out = response.permute(1, 2, 0)
        out = out.cpu().numpy().astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        image_filename = f"{uuid.uuid4()}.png"
        image_path = os.path.join(temp_dir, image_filename)
        cv2.imwrite(image_path, out)

        return chat_history + [(query, "Here is the image generated:"), (None, (image_path,))]
    elif image is not None:
        generation_config = model.default_generation_config
        generation_config.temperature = TEMPERATURE
        generation_config.top_p = TOP_P
        answer = model.generate_content([vila_u.Image(image), query], generation_config)
        media_display = image
    elif video is not None:
        generation_config = model.default_generation_config
        generation_config.temperature = TEMPERATURE
        generation_config.top_p = TOP_P
        answer = model.generate_content([vila_u.Video(video), query], generation_config)
        media_display = video
    else:
        return chat_history + [(None, "No input!")]
    
    return chat_history + [((media_display,), None), (query, answer)]


def clear_chat():
    return None, None, None, []


def regenerate_last_answer(chat_history):
    if len(chat_history) < 1:
        return chat_history 
    
    last_query, last_answer = chat_history[-1]
    if last_query is None:
        if last_answer == "No input!":
            return chat_history
        else:
            return generate_response(None, None, chat_history[-2][0], chat_history[:-2])
    else:
        last_media = chat_history[-2][0][0]
        if is_image_file(last_media):
            return generate_response(last_media, None, last_query, chat_history[:-2])
        else:
            return generate_response(None, last_media, last_query, chat_history[:-2])


def cleanup():
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda' 
    else:
        raise ValueError("CUDA is not available on this machine. Please use a CUDA-enabled machine to run this demo.")
    model = vila_u.load(args.model_path).to(device)

    temp_dir = 'temp/'
    os.makedirs(temp_dir, exist_ok=True)
    
    signal.signal(signal.SIGINT, lambda s, f: (cleanup(), sys.exit()))

    with gr.Blocks(title='VILA-U') as demo:
        gr.Markdown("# VILA-U:  a Unified Foundation Model Integrating Visual Understanding and Generation")
        websites = (
            """
            [[Paper](https://arxiv.org/abs/2409.04429)]
            [[Project](https://hanlab.mit.edu/projects/vila-u)]
            [[GitHub](https://github.com/mit-han-lab/vila-u)] 
            [[Models](https://huggingface.co/collections/mit-han-lab/vila-u-7b-6716f7dd5331e4bdf944ffa6)]
            """
        )
        gr.Markdown(websites)

        with gr.Row():
            with gr.Column(scale=2):
                image_input = gr.Image(label="Upload Image", type="filepath")
                video_input = gr.Video(label="Upload Video", type="filepath")

            with gr.Column(scale=4):
                output_container = gr.Chatbot(
                    label="VILA-U Chatbot",
                    height=400,
                    layout="panel",
                )

        with gr.Row():
            question_input = gr.Textbox(show_label=False, \
                placeholder="Submit a question along with visual input, or provide an image generation prompt alone.", container=False, scale=6)

            submit_button = gr.Button("Submit", variant="primary", scale=1)
            clear_button = gr.Button(value="🗑️  Clear", scale=1)
            retry_button = gr.Button(value="🔄  Retry", scale=1)
        
        with gr.Row():
            gr.Examples(examples=[
                ["assets/example_image1.jpg", "Can you describe what is happening?"],
                ["assets/example_image2.jpg", "What is the brand of the silver car in the image?"],
            ], inputs=[image_input, question_input], cache_examples=False, label="Image Understanding Examples.")

            gr.Examples(examples=[
                ["assets/example_video1.mp4", "Elaborate on the visual and narrative elements of the video in detail."],
                ["assets/example_video2.mp4", "What is the man putting on the plate?"],
            ], inputs=[video_input, question_input], cache_examples=False, label="Video Understanding Examples.")

            gr.Examples(examples=[
                ["An elephant walking in the water."],
                ["A melting apple."],
                ["An astronaut riding a horse on the moon, oil painting by Van Gogh."],
                ["New England fall with leaves, house and river."],
                ["An old man with white beard."],
                ["A crystal tree shimmering under a starry sky."],
                ["A deep forest clearing with a mirrored pond reflecting a galaxyfilled night sky."],
                ["Happy dreamy owl monster sitting on a tree branch, colorful glittering particles, forest background, detailed feathers."]
            ], inputs=[question_input], cache_examples=False, label="Image Generation Examples.")

        submit_button.click(generate_response, inputs=[image_input, video_input, question_input, output_container], outputs=output_container)
        clear_button.click(clear_chat, outputs=[image_input, video_input, question_input, output_container])
        retry_button.click(regenerate_last_answer, inputs=output_container, outputs=output_container)

    try:
        demo.launch(share=True)
    finally:
        cleanup()