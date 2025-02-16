from openai import OpenAI
from datetime import datetime
import json
import logging
from typing import Optional, Dict, Any
import backoff  # For exponential backoff on API errors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PoetryGenerator:
    """A class to handle AI-powered poetry generation with enhanced features"""
    
    def __init__(self, api_key: str, model_version: str = "gpt-4-turbo-preview"):
        self.client = OpenAI(api_key=api_key)
        self.model_version = model_version
        self.default_params = {
            "temperature": 0.7,  # Balance creativity and coherence
            "max_tokens": 500,    # Control poem length
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
        }
        
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate_poem(
        self,
        concept: str,
        style: str = "sonnet",
        technical_depth: int = 2,
        creativity_level: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a technical poem with specified parameters
        
        Args:
            concept: Technical concept to explain
            style: Poetic form (sonnet, haiku, free verse)
            technical_depth: 1 (simple) to 3 (detailed)
            creativity_level: 1 (literal) to 5 (abstract)
            
        Returns:
            Dictionary containing poem and metadata
        """
        try:
            system_prompt = self._build_system_prompt(style, technical_depth, creativity_level)
            
            response = self.client.chat.completions.create(
                model=self.model_version,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Explain {concept} through {style} poetry. Technical depth: {technical_depth}/3, Creativity: {creativity_level}/5"
                    }
                ],
                **self.default_params,
                response_format={"type": "json_object"}  # Request structured output
            )
            
            return self._process_response(response, concept)
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return None

    def _build_system_prompt(self, style: str, depth: int, creativity: int) -> str:
        """Construct detailed system prompt based on parameters"""
        return f"""You are Poet-Engineer, a technical poetry generator. Create a {style} that:
1. Accurately explains complex programming concepts
2. Follows {style} structure rules
3. Uses creative metaphors related to computer science
4. Technical accuracy level: {depth}/3 ({['basic', 'intermediate', 'advanced'][depth-1]})
5. Creativity level: {creativity}/5
6. Include:
   - Core concept definition
   - Common use cases
   - Potential pitfalls
   - Memorable analogy
   - Technical terms in context

Output JSON format with keys: title, stanzas, explanation, metaphors, technical_terms"""

    def _process_response(self, response, concept: str) -> Dict[str, Any]:
        """Validate and format API response"""
        try:
            content = json.loads(response.choices[0].message.content)
            
            # Validate response structure
            required_keys = {"title", "stanzas", "explanation", "metaphors"}
            if not required_keys.issubset(content.keys()):
                raise ValueError("Missing required keys in response")
                
            return {
                "metadata": {
                    "concept": concept,
                    "model": self.model_version,
                    "timestamp": datetime.now().isoformat(),
                    "version": "2.1.0"
                },
                "content": content
            }
            
        except json.JSONDecodeError:
            logger.error("Invalid JSON response")
            return {"error": "Invalid response format"}
            
    def save_to_file(self, poem_data: Dict[str, Any], filename: str) -> None:
        """Save generated poem to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(poem_data, f, indent=2)
            logger.info(f"Poem saved to {filename}")
        except IOError as e:
            logger.error(f"Failed to save file: {str(e)}")

# Example usage
if __name__ == "__main__":
    API_KEY = "your-api-key-here"  # Use environment variables in production
    
    generator = PoetryGenerator(api_key=API_KEY)
    
    poem_data = generator.generate_poem(
        concept="recursion in programming",
        style="villanelle",  # Complex 19-line form
        technical_depth=3,
        creativity_level=4
    )
    
    if poem_data:
        print("=== Technical Poetry ===")
        print(f"\nTitle: {poem_data['content']['title']}\n")
        for stanza in poem_data["content"]["stanzas"]:
            print(stanza + "\n")
            
        print("\n=== Explanation ===")
        print(poem_data["content"]["explanation"])
        
        generator.save_to_file(poem_data, "recursion_poem.json")
    else:
        print("Failed to generate poem")
