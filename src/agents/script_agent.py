from typing import Any, Dict, List
import openai
from .base_agent import BaseAgent

class ScriptAgent(BaseAgent):
    """Agent responsible for generating engaging scripts from historical facts."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "script_agent")
        self.model = config["model"]
        self.max_tokens = config["max_tokens"]
        self.temperature = config["temperature"]
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an engaging script based on the researched facts.
        
        Args:
            input_data: Dictionary containing:
                - facts: List of historical facts
                - platform: Target social media platform
                - style: Content style requirements
                - duration: Target duration in seconds
                
        Returns:
            Dictionary containing:
                - script: The generated script
                - hook: Opening hook for the content
                - call_to_action: Closing call to action
                - metadata: Additional script metadata
        """
        facts = input_data["facts"]
        platform = input_data["platform"]
        style = input_data["style"]
        duration = input_data["duration"]
        
        # Construct the script generation prompt
        prompt = self._construct_script_prompt(facts, platform, style, duration)
        
        # Get script from OpenAI
        response = await self._get_script(prompt)
        
        # Process and structure the script
        script_parts = self._process_script(response)
        
        return {
            "script": script_parts["main_content"],
            "hook": script_parts["hook"],
            "call_to_action": script_parts["call_to_action"],
            "metadata": {
                "platform": platform,
                "style": style,
                "duration": duration,
                "model": self.model
            }
        }
    
    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """
        Validate the script output.
        
        Args:
            output: Dictionary containing the script output
            
        Returns:
            Boolean indicating whether the output is valid
        """
        required_keys = ["script", "hook", "call_to_action", "metadata"]
        if not all(key in output for key in required_keys):
            self.logger.error("Missing required keys in output")
            return False
            
        if not output["script"] or not output["hook"] or not output["call_to_action"]:
            self.logger.error("Empty script components")
            return False
            
        # Verify script length is appropriate for duration
        words = len(output["script"].split())
        if words < 50 or words > 300:  # Adjust these thresholds based on your needs
            self.logger.error("Script length inappropriate")
            return False
            
        return True
    
    def _construct_script_prompt(self, facts: List[str], platform: str, style: str, duration: int) -> str:
        """Construct the script generation prompt for the LLM."""
        facts_text = "\n".join([f"- {fact}" for fact in facts])
        
        return f"""
        Create an engaging script for a {platform} video about these historical facts:
        
        {facts_text}
        
        Requirements:
        - Target platform: {platform}
        - Content style: {style}
        - Duration: {duration} seconds
        - Include a hook that grabs attention
        - Make the content engaging and educational
        - End with a call to action
        
        Format the response as:
        HOOK:
        [Attention-grabbing opening]
        
        MAIN CONTENT:
        [Engaging script that presents the facts]
        
        CALL TO ACTION:
        [Engaging closing statement]
        """
    
    async def _get_script(self, prompt: str) -> str:
        """Get script from OpenAI."""
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a creative scriptwriter specializing in educational content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error getting script from OpenAI: {str(e)}")
            raise
    
    def _process_script(self, response: str) -> Dict[str, str]:
        """Process the script response into structured components."""
        script_parts = {
            "hook": "",
            "main_content": "",
            "call_to_action": ""
        }
        
        current_section = None
        current_content = []
        
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            if line == "HOOK:":
                if current_section:
                    script_parts[current_section] = "\n".join(current_content)
                current_section = "hook"
                current_content = []
            elif line == "MAIN CONTENT:":
                if current_section:
                    script_parts[current_section] = "\n".join(current_content)
                current_section = "main_content"
                current_content = []
            elif line == "CALL TO ACTION:":
                if current_section:
                    script_parts[current_section] = "\n".join(current_content)
                current_section = "call_to_action"
                current_content = []
            else:
                current_content.append(line)
                
        # Add the last section
        if current_section:
            script_parts[current_section] = "\n".join(current_content)
            
        return script_parts 