import argparse
from functools import partial
from pathlib import Path

from diffusers import AutoPipelineForInpainting
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import yaml
from tqdm.auto import tqdm



def save_image(image: np.ndarray, path: Path):
    if image.ndim == 2: # Mask image
        image = Image.fromarray(image.astype(np.uint8) * 255, 'L')
    
    else: # RGB image
        image = Image.fromarray(image.astype(np.uint8), 'RGB')
    
    image.save(path)


def load_sam(model: str, checkpoint: str, device: torch.device) -> SamAutomaticMaskGenerator:
    sam = sam_model_registry[model](checkpoint=checkpoint)
    sam.to(device=device)
    sam.eval()
    mask_generator = SamAutomaticMaskGenerator(sam)

    return mask_generator


def load_clip(model: str, processor: str, device: torch.device) -> tuple[CLIPProcessor, CLIPModel]:
    processor = CLIPProcessor.from_pretrained(processor)
    model = CLIPModel.from_pretrained(model)
    model.to(device=device)
    model.eval()

    return processor, model


def get_masked_images(image: np.ndarray, masks: list[dict[str, np.ndarray | tuple[int, int, int, int]]]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    masked_images = []
    cropped_images = []

    for mask in masks:
        bbox = mask["bbox"]
        segmentation = mask["segmentation"]

        bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        
        cropped_image = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
        cropped_images.append(cropped_image)

        seg_mask = np.expand_dims(segmentation, axis=-1)
        masked_image = ~seg_mask * image + seg_mask * np.array([255, 0, 0])
        masked_images.append(masked_image)
    
    return masked_images, cropped_images


def load_pipeline(config: dict[str, str | float | int], device: torch.device) -> AutoPipelineForInpainting:
    pipe = AutoPipelineForInpainting.from_pretrained(config['model'], torch_dtype=torch.float16, variant='fp16').to(device)
    
    return partial(
        pipe,
        guidance_scale=config['guidance_scale'],
        num_inference_steps=config['num_inference_steps'],
        strength=config['strength']
    )



def add_random_masks(mask: np.ndarray, w: int, p: float) -> np.ndarray:
    """This is just to add some additional masks around the source object for a more realistic inpainting"""

    additional_mask = np.zeros(mask.shape, dtype=bool)
    num_rows, num_cols = mask.shape
    
    for row in range(num_rows):
        for col in range(num_cols):
            if mask[row, col]:
                for _ in range(5):
                    start_row, end_row = max(0, row - w//2), min(num_rows, row + w//2)
                    start_col, end_col = max(0, col - w//2), min(num_cols, col + w//2)
                    additional_mask[start_row:end_row, start_col:end_col] = np.where(np.random.rand(end_row - start_row, end_col - start_col) > p, True, additional_mask[start_row:end_row, start_col:end_col])
    

    return mask | additional_mask
                    



def displace_image(image: np.ndarray, mask: np.ndarray, displacement: tuple[int, int], config: dict[str, int | float]) -> tuple[np.ndarray, np.ndarray]:
    """Displaces the object (given by mask) in the image by the given displacement and returns the new image and new mask"""

    # image : (num_rows, num_cols, 3)
    # mask : (num_rows, num_cols)

    harmonization_width = config['harmonization_width']
    random_mask_probability = config['random_mask_probability']
    random_mask_width = config['random_mask_width']

    x_d, y_d = displacement

    num_rows, num_cols, _ = image.shape
    harmonization_mask = np.zeros((num_rows, num_cols), dtype=bool) # (num_rows, num_cols)
    displaced_image = image.copy() # (num_rows, num_cols, 3)
    new_image_mask = np.zeros((num_rows, num_cols), dtype=bool) # (num_rows, num_cols)

    def interpolate(row: int, col: int):
        for radius in range(1, max(num_rows, num_cols) // 2):
            start_row = max(0, row - radius)
            start_col = max(0, col - radius)
            end_row = min(num_cols, row + radius + 1)
            end_col = min(num_rows, col + radius + 1)

            if np.any(~mask[start_row:end_row, start_col:end_col]):
                num_pixels = np.sum(~mask[start_row:end_row, start_col:end_col])
                return np.sum(np.expand_dims(~mask[start_row:end_row, start_col:end_col], axis=-1) * image[start_row:end_row, start_col:end_col], axis=(0, 1)) // num_pixels
            
            radius += 1
        return np.zeros(3)
    

    def harmonize_pixel(row: int, col: int, width: int):
        start_row = max(0, row - width)
        start_col = max(0, col - width)
        end_row = min(num_rows, row + width + 1)
        end_col = min(num_cols, col + width + 1)
        harmonization_mask[start_row:end_row, start_col:end_col] = True
        new_image_mask[row, col] = True



    # Displace each pixel in the mask by the given displacement and generate the harmonization mask
    progress_bar = tqdm(total=np.sum(mask), desc="Displacing object")
    for row in range(num_rows):
        for col in range(num_cols):
            if not mask[row, col]:
                continue
            
            # Get new coordinates
            new_row = row + y_d
            new_col = col + x_d

            # Make sure the new coordinates are within the image
            new_row = max(0, new_row)
            new_col = max(0, new_col)
            new_row = min(num_rows - 1, new_row)
            new_col = min(num_cols - 1, new_col)
        
            # Displace the pixel
            displaced_image[new_row, new_col] = image[row, col]

            # Interpolate the pixel
            displaced_image[row, col] = interpolate(row, col)

            # Generate the harmonization mask
            harmonize_pixel(new_row, new_col, width=harmonization_width)
            progress_bar.update(1)
    progress_bar.close()


    # Make sure the harmonization mask doesn't contain the displaced object
    harmonization_mask = harmonization_mask & ~new_image_mask

    # Interpolate the pixels in the harmonization mask
    for row in range(num_rows):
        for col in range(num_cols):
            if harmonization_mask[row, col]:
                displaced_image[row, col] = interpolate(row, col)
    

    # Add the original object to the harmonization mask
    harmonization_mask = harmonization_mask | add_random_masks(mask, w=random_mask_width, p=random_mask_probability)

    # Incase some pixels are added to the harmonization mask which contain the displaced object, remove them
    harmonization_mask = harmonization_mask & ~new_image_mask

    return displaced_image, harmonization_mask


@torch.no_grad()
def main(input_path: Path, output_path: Path, prompt: str, displacement: None | tuple[int, int], config: dict[str, dict[str]]):
    # Load the models
    print("Loading models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = load_sam(config['sam']['model'], config['sam']['checkpoint'], device)
    processor, model = load_clip(config['clip']['model'], config['clip']['processor'], device)

    # # Generate the masks
    print("Generating masks...")
    image = np.array(Image.open(input_path))
    masks = sam.generate(image)
    masked_images, cropped_images = get_masked_images(image, masks)

    # Select the mask with the highest similarity score with the prompt
    print(f"Selecting best mask based on prompt: '{prompt}'...")
    clip_inputs = processor(text=prompt, images=cropped_images, return_tensors='pt').to(device)
    similarity_scores: torch.Tensor = model(**clip_inputs).logits_per_image
    best_mask_idx = similarity_scores.squeeze().argmax().cpu().item()
    

    # If displacement is not provided, just save the masked image
    if displacement is None:
        masked_image = masked_images[best_mask_idx]
        save_image(masked_image, output_path)
        print(f"\nMasked image saved at {output_path}")
        return
    

    # Displace the object in the image
    mask = masks[best_mask_idx]['segmentation']
    displaced_image, harmonization_mask = displace_image(image, mask, displacement, config=config['diffusion'])

    # Load the inpainting model
    print("Loading stable diffusion inpainting model...")
    pipeline = load_pipeline(config['diffusion'], device)

    # Inpaint the displaced image
    displaced_image = Image.fromarray(displaced_image.astype(np.uint8), 'RGB')
    harmonization_mask = Image.fromarray(harmonization_mask.astype(np.uint8) * 255, 'L')
    harmonized_image: Image = pipeline(prompt=config['diffusion']['background_prompt'], image=displaced_image, mask_image=harmonization_mask).images[0]
    harmonized_image.save(output_path)

    print(f"\nFinal image saved at {output_path}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Object Displacement Program')

    parser.add_argument('--image', dest='input_path', type=str, required=True, help='Path to the image which requires editing')
    parser.add_argument('--output', dest='output_path', type=str, required=True, help='Path where the edited image will be saved')
    parser.add_argument('--class', dest='prompt', type=str, required=True, help='Text prompt for the class of the object to be displaced')
    parser.add_argument('--x', type=int, default=None, help='Displacement in the x-axis. If not provided, the program will just print the masked image')
    parser.add_argument('--y', type=int, default=None, help='Displacement in the y-axis. If not provided, the program will just print the masked image')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file. Defaults to config.yaml')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    # Validate the arguments
    if args.x is not None and args.y is None or args.y is not None and args.x is None:
        parser.error('Both x and y must be provided for displacement')
    
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    prompt = args.prompt
    displacement = (args.x, -args.y) if args.x is not None else None # Negative cause the top rows have lower indices
    config_path = Path(args.config)
    
    # Make sure the input image exists
    if not input_path.exists():
        parser.error(f"Input image not found at {input_path}")
    
    # Make sure the config file exists
    if not config_path.exists():
        parser.error(f"Config file not found at {config_path}")
    
    # Create the output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the config file
    config = yaml.full_load(config_path.open())

    main(input_path, output_path, prompt, displacement, config)
