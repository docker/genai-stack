from typing import Any, Dict
import os
from google.cloud import texttospeech
from .base_agent import BaseAgent

class VoiceAgent(BaseAgent):
    """Agent responsible for generating voice-overs using Google Cloud Text-to-Speech."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "voice_agent")
        self.provider = config["provider"]
        self.voice_id = config["voice_id"]
        self.speaking_rate = config["speaking_rate"]
        self.client = texttospeech.TextToSpeechClient()
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate voice-over audio for the script.
        
        Args:
            input_data: Dictionary containing:
                - script: The script to convert to speech
                - platform: Target social media platform
                - style: Content style requirements
                
        Returns:
            Dictionary containing:
                - audio_path: Path to the generated audio file
                - duration: Duration of the audio in seconds
                - metadata: Additional voice-over metadata
        """
        script = input_data["script"]
        platform = input_data["platform"]
        style = input_data["style"]
        
        # Configure voice settings
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name=self.voice_id,
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        
        # Configure audio settings
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=self.speaking_rate,
            pitch=0.0
        )
        
        # Generate SSML with appropriate pauses and emphasis
        ssml = self._generate_ssml(script, style)
        
        # Synthesize speech
        synthesis_input = texttospeech.SynthesisInput(ssml=ssml)
        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Save the audio file
        output_dir = "output/audio"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/voiceover_{platform}_{hash(script)}.mp3"
        
        with open(output_file, "wb") as out:
            out.write(response.audio_content)
            
        return {
            "audio_path": output_file,
            "duration": len(response.audio_content) / 16000,  # Approximate duration
            "metadata": {
                "platform": platform,
                "style": style,
                "voice_id": self.voice_id,
                "speaking_rate": self.speaking_rate
            }
        }
    
    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """
        Validate the voice-over output.
        
        Args:
            output: Dictionary containing the voice-over output
            
        Returns:
            Boolean indicating whether the output is valid
        """
        required_keys = ["audio_path", "duration", "metadata"]
        if not all(key in output for key in required_keys):
            self.logger.error("Missing required keys in output")
            return False
            
        if not os.path.exists(output["audio_path"]):
            self.logger.error("Audio file not created")
            return False
            
        if output["duration"] <= 0:
            self.logger.error("Invalid audio duration")
            return False
            
        return True
    
    def _generate_ssml(self, script: str, style: str) -> str:
        """Generate SSML with appropriate pauses and emphasis."""
        # Split script into sentences
        sentences = [s.strip() for s in script.split(".") if s.strip()]
        
        # Add SSML tags for each sentence
        ssml_parts = []
        for sentence in sentences:
            # Add emphasis to key words (you might want to make this more sophisticated)
            words = sentence.split()
            if len(words) > 5:  # Only emphasize in longer sentences
                # Emphasize the first and last significant words
                words[0] = f"<emphasis level='moderate'>{words[0]}</emphasis>"
                words[-1] = f"<emphasis level='moderate'>{words[-1]}</emphasis>"
            
            ssml_parts.append(" ".join(words))
            
        # Combine with appropriate pauses
        ssml = f"""
        <speak>
            {"<break time='500ms'/>".join(ssml_parts)}
        </speak>
        """
        
        return ssml 