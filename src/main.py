import asyncio
import yaml
from typing import Dict, Any
from agents.research_agent import ResearchAgent
from agents.script_agent import ScriptAgent
from agents.voice_agent import VoiceAgent
from agents.visual_agent import VisualAgent
from agents.editor_agent import EditorAgent

class ContentGenerator:
    """Main orchestrator for the content generation pipeline."""
    
    def __init__(self, config_path: str = "src/config/content_config.yaml"):
        """Initialize the content generator with configuration."""
        self.config = self._load_config(config_path)
        self.agents = self._initialize_agents()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents with their configurations."""
        return {
            "research": ResearchAgent(self.config["agents"]["research"]),
            "script": ScriptAgent(self.config["agents"]["script"]),
            "voice": VoiceAgent(self.config["agents"]["voice"]),
            "visual": VisualAgent(self.config["agents"]["visual"]),
            "editor": EditorAgent(self.config["agents"]["editor"])
        }
        
    async def generate_content(self, topic: str, platform: str) -> Dict[str, Any]:
        """
        Generate content for a specific topic and platform.
        
        Args:
            topic: The historical topic to create content about
            platform: Target social media platform
            
        Returns:
            Dictionary containing the generated content and metadata
        """
        try:
            # Get platform-specific settings
            platform_config = self.config["platforms"][platform]
            
            # Step 1: Research
            research_input = {
                "topic": topic,
                "platform": platform,
                "style": platform_config["style"]
            }
            research_output = await self.agents["research"].run(research_input)
            if not research_output:
                raise Exception("Research phase failed")
                
            # Step 2: Script Generation
            script_input = {
                "facts": research_output["facts"],
                "platform": platform,
                "style": platform_config["style"],
                "duration": platform_config["max_duration"]
            }
            script_output = await self.agents["script"].run(script_input)
            if not script_output:
                raise Exception("Script generation failed")
                
            # Step 3: Voice-over Generation
            voice_input = {
                "script": script_output["script"],
                "platform": platform,
                "style": platform_config["style"]
            }
            voice_output = await self.agents["voice"].run(voice_input)
            if not voice_output:
                raise Exception("Voice-over generation failed")
                
            # Step 4: Visual Asset Generation
            visual_input = {
                "script": script_output["script"],
                "platform": platform,
                "style": platform_config["style"],
                "aspect_ratio": platform_config["aspect_ratio"]
            }
            visual_output = await self.agents["visual"].run(visual_input)
            if not visual_output:
                raise Exception("Visual asset generation failed")
                
            # Step 5: Video Editing
            editor_input = {
                "image_paths": visual_output["image_paths"],
                "audio_path": voice_output["audio_path"],
                "platform": platform,
                "duration": platform_config["max_duration"],
                "aspect_ratio": platform_config["aspect_ratio"]
            }
            editor_output = await self.agents["editor"].run(editor_input)
            if not editor_output:
                raise Exception("Video editing failed")
                
            return {
                "video_path": editor_output["video_path"],
                "metadata": {
                    "topic": topic,
                    "platform": platform,
                    "research": research_output["metadata"],
                    "script": script_output["metadata"],
                    "voice": voice_output["metadata"],
                    "visual": visual_output["metadata"],
                    "editor": editor_output["metadata"]
                }
            }
            
        except Exception as e:
            print(f"Error generating content: {str(e)}")
            return None

async def main():
    """Main entry point for the content generation system."""
    # Example usage
    generator = ContentGenerator()
    
    # Generate content for different platforms
    platforms = ["tiktok", "youtube_shorts", "instagram", "facebook"]
    topic = "Ancient Egyptian Pyramids"
    
    for platform in platforms:
        print(f"\nGenerating content for {platform}...")
        result = await generator.generate_content(topic, platform)
        
        if result:
            print(f"Successfully generated content for {platform}")
            print(f"Video saved at: {result['video_path']}")
        else:
            print(f"Failed to generate content for {platform}")

if __name__ == "__main__":
    asyncio.run(main()) 