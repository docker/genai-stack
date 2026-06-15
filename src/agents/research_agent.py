from typing import Any, Dict, List
import ollama
from .base_agent import BaseAgent

class ResearchAgent(BaseAgent):
    """Agent responsible for researching and verifying historical facts."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "research_agent")
        self.model = config["model"]  # misalnya "llama2"
        self.max_tokens = config["max_tokens"]
        self.temperature = config["temperature"]
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Research and verify historical facts based on the input topic.
        
        Args:
            input_data: Dictionary containing:
                - topic: The historical topic to research
                - platform: Target social media platform
                - style: Content style requirements
                
        Returns:
            Dictionary containing:
                - facts: List of verified historical facts
                - sources: List of sources for verification
                - metadata: Additional research metadata
        """
        topic = input_data["topic"]
        platform = input_data["platform"]
        style = input_data["style"]
        
        # Construct the research prompt
        prompt = self._construct_research_prompt(topic, platform, style)
        
        # Get research from Ollama
        response = await self._get_research(prompt)
        
        # Process and structure the research
        facts, sources = self._process_research(response)
        
        return {
            "facts": facts,
            "sources": sources,
            "metadata": {
                "topic": topic,
                "platform": platform,
                "style": style,
                "model": self.model
            }
        }
    
    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """
        Validate the research output.
        
        Args:
            output: Dictionary containing the research output
            
        Returns:
            Boolean indicating whether the output is valid
        """
        required_keys = ["facts", "sources", "metadata"]
        if not all(key in output for key in required_keys):
            self.logger.error("Missing required keys in output")
            return False
            
        if not output["facts"] or not output["sources"]:
            self.logger.error("Empty facts or sources")
            return False
            
        # Verify that each fact has at least one source
        if len(output["facts"]) > len(output["sources"]):
            self.logger.error("More facts than sources")
            return False
            
        return True
    
    def _construct_research_prompt(self, topic: str, platform: str, style: str) -> str:
        """Construct the research prompt for the LLM."""
        return f"""
        Research the following historical topic: {topic}
        
        Requirements:
        - Target platform: {platform}
        - Content style: {style}
        - Find 3-5 interesting and verified historical facts
        - Include specific dates and details
        - Ensure facts are engaging for social media
        - Provide reliable sources for each fact
        
        Format the response as:
        FACTS:
        - [Fact 1]
        - [Fact 2]
        ...
        
        SOURCES:
        - [Source 1]
        - [Source 2]
        ...
        """
    
    async def _get_research(self, prompt: str) -> str:
        """Get research from Ollama."""
        try:
            response = ollama.generate(self.model, prompt)
            return response['response']
        except Exception as e:
            self.logger.error(f"Error getting research from Ollama: {str(e)}")
            raise
    
    def _process_research(self, response: str) -> tuple[List[str], List[str]]:
        """Process the research response into structured facts and sources."""
        facts = []
        sources = []
        
        current_section = None
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            if line == "FACTS:":
                current_section = "facts"
            elif line == "SOURCES:":
                current_section = "sources"
            elif line.startswith("- "):
                if current_section == "facts":
                    facts.append(line[2:])
                elif current_section == "sources":
                    sources.append(line[2:])
        return facts, sources 