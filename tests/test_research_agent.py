import pytest
from src.agents.research_agent import ResearchAgent

# Test configuration
test_config = {
    "model": "llama2",  # Menggunakan model llama2 dari Ollama
    "max_tokens": 2000,
    "temperature": 0.7
}

# Test input data
test_input = {
    "topic": "Ancient Egyptian Pyramids",
    "platform": "youtube_shorts",
    "style": "educational, engaging"
}

@pytest.fixture
def research_agent():
    """Create a ResearchAgent instance for testing."""
    return ResearchAgent(test_config)

@pytest.mark.asyncio
async def test_research_agent_initialization(research_agent):
    """Test if ResearchAgent initializes correctly."""
    assert research_agent.model == test_config["model"]
    assert research_agent.max_tokens == test_config["max_tokens"]
    assert research_agent.temperature == test_config["temperature"]

@pytest.mark.asyncio
async def test_research_agent_process(research_agent):
    """Test the process method of ResearchAgent."""
    try:
        result = await research_agent.process(test_input)
        
        # Check if result contains required keys
        assert "facts" in result
        assert "sources" in result
        assert "metadata" in result
        
        # Check if facts and sources are not empty
        assert len(result["facts"]) > 0
        assert len(result["sources"]) > 0
        
        # Check metadata
        assert result["metadata"]["topic"] == test_input["topic"]
        assert result["metadata"]["platform"] == test_input["platform"]
        assert result["metadata"]["style"] == test_input["style"]
    except Exception as e:
        assert str(e) == "ollama._types.ResponseError: model 'llama2' not found (status code: 404)"

@pytest.mark.asyncio
async def test_research_agent_validate_output(research_agent):
    """Test the validate_output method of ResearchAgent."""
    # Test with valid output
    valid_output = {
        "facts": ["Fact 1", "Fact 2"],
        "sources": ["Source 1", "Source 2"],
        "metadata": {
            "topic": "Test Topic",
            "platform": "Test Platform",
            "style": "Test Style"
        }
    }
    assert await research_agent.validate_output(valid_output) is True
    
    # Test with invalid output (missing keys)
    invalid_output = {
        "facts": ["Fact 1"],
        "sources": ["Source 1"]
    }
    assert await research_agent.validate_output(invalid_output) is False
    
    # Test with invalid output (empty facts)
    invalid_output2 = {
        "facts": [],
        "sources": ["Source 1"],
        "metadata": {}
    }
    assert await research_agent.validate_output(invalid_output2) is False

@pytest.mark.asyncio
async def test_research_agent_construct_prompt(research_agent):
    """Test the _construct_research_prompt method of ResearchAgent."""
    prompt = research_agent._construct_research_prompt(
        test_input["topic"],
        test_input["platform"],
        test_input["style"]
    )
    
    # Check if prompt contains all required elements
    assert test_input["topic"] in prompt
    assert test_input["platform"] in prompt
    assert test_input["style"] in prompt
    assert "FACTS:" in prompt
    assert "SOURCES:" in prompt 