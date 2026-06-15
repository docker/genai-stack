from typing import Any, Dict, List
import os
from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    AudioFileClip,
    CompositeVideoClip,
    concatenate_videoclips
)
from .base_agent import BaseAgent

class EditorAgent(BaseAgent):
    """Agent responsible for combining audio and visual assets into the final video."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "editor_agent")
        self.video_quality = config["video_quality"]
        self.audio_quality = config["audio_quality"]
        self.transition_style = config["transition_style"]
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine audio and visual assets into the final video.
        
        Args:
            input_data: Dictionary containing:
                - image_paths: List of paths to visual assets
                - audio_path: Path to the voice-over audio
                - platform: Target social media platform
                - duration: Target duration in seconds
                - aspect_ratio: Required aspect ratio
                
        Returns:
            Dictionary containing:
                - video_path: Path to the generated video
                - metadata: Additional video metadata
        """
        image_paths = input_data["image_paths"]
        audio_path = input_data["audio_path"]
        platform = input_data["platform"]
        duration = input_data["duration"]
        aspect_ratio = input_data["aspect_ratio"]
        
        # Create video clips from images
        video_clips = self._create_video_clips(image_paths, duration)
        
        # Add transitions between clips
        video_clips = self._add_transitions(video_clips)
        
        # Combine video clips
        final_video = concatenate_videoclips(video_clips)
        
        # Add audio
        audio = AudioFileClip(audio_path)
        final_video = final_video.set_audio(audio)
        
        # Save the final video
        output_dir = "output/videos"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{platform}_final.mp4"
        
        final_video.write_videofile(
            output_file,
            fps=30,
            codec='libx264',
            audio_codec='aac',
            preset='medium'
        )
        
        return {
            "video_path": output_file,
            "metadata": {
                "platform": platform,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "video_quality": self.video_quality,
                "audio_quality": self.audio_quality,
                "transition_style": self.transition_style
            }
        }
    
    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """
        Validate the video output.
        
        Args:
            output: Dictionary containing the video output
            
        Returns:
            Boolean indicating whether the output is valid
        """
        required_keys = ["video_path", "metadata"]
        if not all(key in output for key in required_keys):
            self.logger.error("Missing required keys in output")
            return False
            
        if not os.path.exists(output["video_path"]):
            self.logger.error("Video file not created")
            return False
            
        try:
            video = VideoFileClip(output["video_path"])
            if video.duration <= 0:
                self.logger.error("Invalid video duration")
                return False
            video.close()
        except Exception as e:
            self.logger.error(f"Invalid video file: {str(e)}")
            return False
            
        return True
    
    def _create_video_clips(self, image_paths: List[str], total_duration: float) -> List[VideoFileClip]:
        """Create video clips from images with appropriate durations."""
        clips = []
        duration_per_clip = total_duration / len(image_paths)
        
        for image_path in image_paths:
            clip = ImageClip(image_path).set_duration(duration_per_clip)
            clips.append(clip)
            
        return clips
    
    def _add_transitions(self, clips: List[VideoFileClip]) -> List[VideoFileClip]:
        """Add transitions between video clips."""
        if self.transition_style == "smooth":
            # Add crossfade transitions
            for i in range(len(clips) - 1):
                clips[i] = clips[i].crossfadein(0.5)
                clips[i + 1] = clips[i + 1].crossfadein(0.5)
        elif self.transition_style == "fade":
            # Add fade transitions
            for i in range(len(clips)):
                clips[i] = clips[i].fadein(0.5).fadeout(0.5)
                
        return clips 