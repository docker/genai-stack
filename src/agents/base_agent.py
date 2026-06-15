from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
from pathlib import Path

class BaseAgent(ABC):
    """Base class for all AI agents in the content generation system."""
    
    def __init__(self, config: Dict[str, Any], name: str):
        """
        Initialize the base agent.
        
        Args:
            config: Configuration dictionary for the agent
            name: Name of the agent
        """
        self.config = config
        self.name = name
        self.logger = logging.getLogger(f"agent.{name}")
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging for the agent."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        handler = logging.FileHandler(log_dir / f"{self.name}.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data and return the result.
        
        Args:
            input_data: Dictionary containing input data for the agent
            
        Returns:
            Dictionary containing the processed output
        """
        pass
    
    @abstractmethod
    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """
        Validate the output of the agent.
        
        Args:
            output: Dictionary containing the output to validate
            
        Returns:
            Boolean indicating whether the output is valid
        """
        pass
    
    async def run(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Run the agent's processing pipeline.
        
        Args:
            input_data: Dictionary containing input data for the agent
            
        Returns:
            Dictionary containing the processed output, or None if processing failed
        """
        try:
            self.logger.info(f"Starting processing with input: {input_data}")
            output = await self.process(input_data)
            
            if await self.validate_output(output):
                self.logger.info("Processing completed successfully")
                return output
            else:
                self.logger.error("Output validation failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Error during processing: {str(e)}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent.
        
        Returns:
            Dictionary containing the agent's status information
        """
        return {
            "name": self.name,
            "status": "active",
            "config": self.config
        } 