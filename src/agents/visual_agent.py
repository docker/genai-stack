from typing import Any, Dict, List
import os
import openai
from PIL import Image
import requests
from io import BytesIO
from .base_agent import BaseAgent

class VisualAgent(BaseAgent):
    """Agent responsible for generating or selecting visual assets using DALL-E 3."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "visual_agent")
        self.provider = config["provider"]
        self.style = config["style"]
        self.image_quality = config["image_quality"]
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate or select visual assets based on the script and platform requirements.
        
        Args:
            input_data: Dictionary containing:
                - script: The script to generate visuals for
                - platform: Target social media platform
                - style: Content style requirements
                - aspect_ratio: Required aspect ratio for the platform
                
        Returns:
            Dictionary containing:
                - image_paths: List of paths to generated images
                - metadata: Additional visual asset metadata
        """
        script = input_data["script"]
        platform = input_data["platform"]
        style = input_data["style"]
        aspect_ratio = input_data["aspect_ratio"]
        
        # Generate image prompts from script
        prompts = self._generate_image_prompts(script, style)
        
        # Generate images for each prompt
        image_paths = []
        for i, prompt in enumerate(prompts):
            image_path = await self._generate_image(
                prompt,
                platform,
                aspect_ratio,
                f"image_{i}"
            )
            image_paths.append(image_path)
            
        return {
            "image_paths": image_paths,
            "metadata": {
                "platform": platform,
                "style": style,
                "aspect_ratio": aspect_ratio,
                "provider": self.provider
            }
        }
    
    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """
        Validate the visual asset output.
        
        Args:
            output: Dictionary containing the visual asset output
            
        Returns:
            Boolean indicating whether the output is valid
        """
        required_keys = ["image_paths", "metadata"]
        if not all(key in output for key in required_keys):
            self.logger.error("Missing required keys in output")
            return False
            
        if not output["image_paths"]:
            self.logger.error("No images generated")
            return False
            
        # Verify all images exist and are valid
        for image_path in output["image_paths"]:
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return False
            try:
                Image.open(image_path)
            except Exception as e:
                self.logger.error(f"Invalid image file {image_path}: {str(e)}")
                return False
                
        return True
    
    def _generate_image_prompts(self, script: str, style: str) -> List[str]:
        """Generate image prompts from the script."""
        # Split script into key moments
        sentences = [s.strip() for s in script.split(".") if s.strip()]
        
        # Generate prompts for key moments
        prompts = []
        for sentence in sentences:
            prompt = f"""
            Create a {style} image that illustrates: {sentence}
            Style: {self.style}
            Quality: {self.image_quality}
            Make it visually appealing and historically accurate.
            """
            prompts.append(prompt)
            
        return prompts
    
    async def _generate_image(self, prompt: str, platform: str, aspect_ratio: str, image_id: str) -> str:
        """Generate an image using DALL-E 3."""
        try:
            # Configure image generation parameters
            size = self._get_image_size(aspect_ratio)
            
            response = await openai.Image.acreate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality="standard",
                n=1
            )
            
            # Download and save the image
            image_url = response.data[0].url
            image_response = requests.get(image_url)
            image = Image.open(BytesIO(image_response.content))
            
            # Save the image
            output_dir = "output/images"
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/{platform}_{image_id}.png"
            
            image.save(output_file)
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error generating image: {str(e)}")
            raise
    
    def _get_image_size(self, aspect_ratio: str) -> str:
        """Get the appropriate image size based on aspect ratio."""
        size_map = {
            "1:1": "1024x1024",
            "16:9": "1024x576",
            "9:16": "576x1024",
            "4:3": "1024x768",
            "3:4": "768x1024"
        }
        return size_map.get(aspect_ratio, "1024x1024") 