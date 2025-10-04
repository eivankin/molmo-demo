import logging
import os
import re
from enum import StrEnum

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import gradio as gr
import numpy as np
import requests
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE = os.getenv("DEVICE", "cpu")
logger.info(f"Using device: {DEVICE}")


def load_model():
    """Load the model and processor"""
    logger.info("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        "allenai/MolmoE-1B-0924",
        trust_remote_code=True,
        device_map=DEVICE,
    )

    logger.info("Loading model... This might take some time...")
    model = AutoModelForCausalLM.from_pretrained(
        "allenai/MolmoE-1B-0924",
        trust_remote_code=True,
        device_map=DEVICE,
    )

    return processor, model


def draw_pointing_marker(
    image, coordinates, marker_size=20, line_width=3, marker_color="red"
):
    """
    Draw a pointing marker on the image at the specified coordinates
    """
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)

    for point in coordinates:
        if point is None or len(point) != 2:
            return image

        x, y = point

        # Ensure coordinates are within image bounds
        width, height = image.size
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))

        # Horizontal line
        draw.line(
            [(x - marker_size, y), (x + marker_size, y)],
            fill=marker_color,
            width=line_width,
        )
        # Vertical line
        draw.line(
            [(x, y - marker_size), (x, y + marker_size)],
            fill=marker_color,
            width=line_width,
        )
        # Outer circle
        draw.ellipse(
            [
                x - marker_size // 2,
                y - marker_size // 2,
                x + marker_size // 2,
                y + marker_size // 2,
            ],
            outline=marker_color,
            width=line_width,
        )

    return vis_image


def extract_pointing_coordinates(output_string, image):
    """
    Function to get x, y coordinates given Molmo model outputs.
    :param output_string: Output from the Molmo model.
    :param image: Image in PIL format.
    Returns:
    coordinates: Coordinates in format of [(x, y), (x, y)]
    """
    image = np.array(image)
    h, w = image.shape[:2]
    coordinates = None
    if "points" in output_string:
        matches = re.findall(r'(x\d+)="([\d.]+)" (y\d+)="([\d.]+)"', output_string)
        coordinates = [
            (int(float(x_val) / 100 * w), int(float(y_val) / 100 * h))
            for _, x_val, _, y_val in matches
        ]
    else:
        match = re.search(r'x="([\d.]+)" y="([\d.]+)"', output_string)
        if match:
            coordinates = [
                (
                    int(float(match.group(1)) / 100 * w),
                    int(float(match.group(2)) / 100 * h),
                )
            ]
    return coordinates


processor, model = load_model()
logger.info("Model ready!")


class PointingDemoImages(StrEnum):
    """Examples from pixmo-points-eval"""

    CAT = (
        "https://cdn.thewirecutter.com/wp-content/uploads/2018/04/catbeds-lowres-2.jpg"
    )
    CAR_ICONS = (
        "https://i.pinimg.com/originals/2f/b3/9b/2fb39ba5f7074cd409c1dfd45db4b1b6.jpg"
    )
    DOG = "https://www.smallcitybigpersonality.co.uk/pubd/images/upd/164f532709a-20180606-130647.700.jpg"

    def get_prompt(self) -> str:
        match self:
            case PointingDemoImages.CAT:
                return "cat"
            case PointingDemoImages.CAR_ICONS:
                return "car icons"
            case PointingDemoImages.DOG:
                return "white large long haired dog"
            case _:
                raise ValueError(f"No prompt defined for {self}")


class DescriptionDemoImages(StrEnum):
    """Examples from pixmo-cap"""

    MEME = "https://i.redd.it/l5lun7441y3b1.jpg"
    MINECRAFT = "https://i.redd.it/6nj8ny4hnnf21.jpg"
    CAT = "https://i.redd.it/vi3py463o04a1.jpg"


def create_demo_image(image_url: PointingDemoImages | DescriptionDemoImages):
    """Create a sample image for the demo"""
    img = Image.open(requests.get(image_url, stream=True).raw)
    return img


def generate_response(image, prompt, task_type):
    """Generate response from the model based on task type"""
    try:
        inputs = processor.process(images=[image], text=prompt)

        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer,
        )

        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        generated_text = processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        if task_type == "pointing":
            coordinates = extract_pointing_coordinates(generated_text, image)
            return generated_text, coordinates
        else:
            return generated_text, None

    except Exception as e:
        return f"Error: {str(e)}", None


def pointing_demo(image, object_description):
    """Handle pointing task - identify and locate objects in images"""
    if image is None:
        return "Please upload an image first.", None, None

    prompt = f"Point to the {object_description} in this image."
    result, coordinates = generate_response(image, prompt, "pointing")

    if coordinates and image:
        vis_image = draw_pointing_marker(image, coordinates)
        return result, f"Coordinates: {coordinates}", vis_image
    else:
        return result, "No coordinates detected", image


def description_demo(image):
    """Handle image description task"""
    if image is None:
        return "Please upload an image first."

    prompt = "Describe this image in detail."
    result, _ = generate_response(image, prompt, "description")
    return result


pointing_examples = [
    [create_demo_image(demo_image), demo_image.get_prompt()]
    for demo_image in PointingDemoImages
]

description_examples = [
    [create_demo_image(demo_image)] for demo_image in DescriptionDemoImages
]

with gr.Blocks(title="Molmo Multi-Modal Demo") as demo:
    gr.Markdown(
        """
    # 🎯 Molmo Multi-Modal Demo
    **A lightweight vision-language model for pointing and image description tasks**

    This demo showcases two capabilities of the Molmo model:
    - **Pointing**: Identify and locate objects in images
    - **Image Description**: Generate detailed descriptions of images
    """
    )

    with gr.Tabs():
        with gr.TabItem("🎯 Pointing"):
            with gr.Row():
                with gr.Column():
                    pointing_image = gr.Image(
                        label="Upload Image",
                        type="pil",
                        value=create_demo_image(PointingDemoImages.CAT),
                    )
                    pointing_prompt = gr.Textbox(
                        label="Object to Point To",
                        placeholder="e.g., 'dog', 'car', 'red object', 'largest item'",
                        value=PointingDemoImages.CAT.get_prompt(),
                    )
                    pointing_button = gr.Button("Point to Object")

                with gr.Column():
                    pointing_output = gr.Textbox(
                        label="Model Response", lines=4, interactive=False
                    )
                    coordinates_output = gr.Textbox(
                        label="Detected Coordinates", lines=2, interactive=False
                    )
                    visualization_output = gr.Image(
                        label="Image with Pointing Marker",
                        type="pil",
                        interactive=False,
                    )

            pointing_button.click(
                pointing_demo,
                inputs=[pointing_image, pointing_prompt],
                outputs=[pointing_output, coordinates_output, visualization_output],
            )

            gr.Examples(
                examples=pointing_examples,
                inputs=[pointing_image, pointing_prompt],
                label="Quick Examples",
            )

        with gr.TabItem("📝 Image Description"):
            with gr.Row():
                with gr.Column():
                    desc_image = gr.Image(
                        label="Upload Image",
                        type="pil",
                        value=create_demo_image(DescriptionDemoImages.CAT),
                    )
                    desc_button = gr.Button("Describe Image")

                with gr.Column():
                    desc_output = gr.Textbox(
                        label="Image Description", lines=8, interactive=False
                    )

            desc_button.click(
                description_demo, inputs=[desc_image], outputs=[desc_output]
            )

            gr.Examples(
                examples=description_examples,
                inputs=[desc_image],
                label="Quick Examples",
            )

    gr.Markdown(
        """
    ---
    **Model Information:**
    - Model: [MolmoE-1B-0924 from AllenAI](https://huggingface.co/allenai/MolmoE-1B-0924)
    - Technical report: [ArXiv](https://arxiv.org/abs/2409.17146)
    
    **Datasets for evaluation:**
    - [PixMo Pointing](https://huggingface.co/datasets/allenai/pixmo-points-eval)
    - [PixMo Image Captioning](https://huggingface.co/datasets/allenai/pixmo-cap)
    """
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
