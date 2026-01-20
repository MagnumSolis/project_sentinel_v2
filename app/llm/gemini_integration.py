"""
Project Sentinel V2 - Gemini Integration

Wrapper for Google Gemini API with offline fallback support.
Handles both text generation and vision analysis.
"""

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Optional, List, Dict, Any
from pathlib import Path
from PIL import Image
import logging
import json
import re

logger = logging.getLogger(__name__)


class GeminiIntegration:
    """
    Wrapper for Gemini API integration with fallback support.
    
    Features:
    - Text generation with RAG context
    - Image analysis for damage assessment
    - Offline template-based fallback
    - Citation formatting
    """
    
    SYSTEM_PROMPT = """You are Project Sentinel, an AI assistant specialized in disaster response and humanitarian assistance.

Your role is to analyze disaster-related information and provide actionable insights to rescue teams.

When responding:
1. Be concise and prioritize actionable information
2. Always cite your sources using [Source N] format
3. Highlight critical findings (trapped civilians, severe damage, etc.)
4. Provide confidence levels when making assessments
5. Suggest recommended actions when appropriate

Focus on: damage assessment, rescue priorities, resource allocation, and situational awareness."""

    DAMAGE_ANALYSIS_PROMPT = """Analyze this disaster-related image and extract:

1. **Damage Type**: What kind of damage is visible? (building collapse, flooding, fire damage, infrastructure damage, etc.)
2. **Severity Score**: Rate the damage from 0.0 (no damage) to 1.0 (complete destruction)
3. **Key Observations**: What specific elements show damage?
4. **Location Clues**: Any visible landmarks, street signs, or geographic features?
5. **Rescue Priority**: Are there signs of people needing rescue? (1-10 scale, 10 = highest priority)

Respond in JSON format:
{
    "damage_score": 0.0-1.0,
    "damage_categories": ["category1", "category2"],
    "description": "Brief description of damage",
    "key_observations": ["observation1", "observation2"],
    "location_clues": "Any identifiable location information",
    "rescue_priority": 1-10,
    "recommended_action": "Suggested response action"
}"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        vision_model: str = "gemini-2.0-flash",
        offline_mode: bool = False
    ):
        """
        Initialize Gemini Integration.
        
        Args:
            api_key: Google Gemini API key
            model: Model name for text generation
            vision_model: Model name for vision analysis
            offline_mode: If True, skip API calls and use templates
        """
        self.api_key = api_key
        self.model_name = model
        self.vision_model_name = vision_model
        self.offline_mode = offline_mode
        
        self.available = False
        self.text_model = None
        self.vision_model = None
        
        if not offline_mode and api_key:
            try:
                genai.configure(api_key=api_key)
                
                # Configure safety settings (allow disaster-related content)
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
                
                # Initialize models
                self.text_model = genai.GenerativeModel(
                    model_name=model,
                    safety_settings=safety_settings,
                    system_instruction=self.SYSTEM_PROMPT
                )
                
                self.vision_model = genai.GenerativeModel(
                    model_name=vision_model,
                    safety_settings=safety_settings
                )
                
                self.available = True
                logger.info(f"Gemini API initialized with model: {model}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini API: {e}")
                self.available = False
        else:
            logger.info("Running in offline mode (Gemini disabled)")
    
    def generate_response(
        self,
        query: str,
        context: str,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate a grounded response using RAG context.
        
        Args:
            query: User's question
            context: Retrieved context from vector search
            max_tokens: Maximum response tokens
        
        Returns:
            Generated response with citations
        """
        if not self.available or self.offline_mode:
            return self._generate_offline_response(query, context)
        
        try:
            # Construct prompt with context
            prompt = f"""Based on the following context, answer the user's question.

CONTEXT:
{context}

USER QUESTION: {query}

Provide a comprehensive answer citing relevant sources. If the context doesn't contain enough information, acknowledge this clearly."""

            # Generate response
            response = self.text_model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0.7
                }
            )
            
            if response.text:
                return response.text
            else:
                logger.warning("Empty response from Gemini")
                return self._generate_offline_response(query, context)
                
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return self._generate_offline_response(query, context)
    
    def _generate_offline_response(
        self,
        query: str,
        context: str
    ) -> str:
        """
        Generate a template-based response when offline.
        
        Args:
            query: User's question
            context: Retrieved context
        
        Returns:
            Formatted response based on context
        """
        if not context or context == "No relevant information found in the database.":
            return f"""**Query:** {query}

**Status:** No relevant information found in the database.

**Recommendation:** 
- Upload relevant disaster reports, satellite imagery, or emergency call recordings
- Try rephrasing your query with different keywords
- Check if the Qdrant database has been populated with sample data"""
        
        # Parse context to extract sources
        sources = context.split("[Source")
        num_sources = len(sources) - 1
        
        response = f"""**Query:** {query}

**Analysis Summary (Offline Mode):**

Based on {num_sources} source(s) found in the database:

{context[:2000]}{'...' if len(context) > 2000 else ''}

---

**Note:** This is an offline response. For enhanced AI-powered analysis:
1. Set `OFFLINE_MODE=False` in your `.env` file
2. Provide a valid `GEMINI_API_KEY`
3. Restart the application

**Recommended Actions:**
- Review the sources above for detailed information
- Cross-reference multiple sources for verification
- Deploy field teams to high-priority areas identified"""
        
        return response
    
    def analyze_image(
        self,
        image_path: str
    ) -> Dict[str, Any]:
        """
        Analyze an image for damage assessment.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dict with damage_score, damage_categories, description, etc.
        """
        default_result = {
            "damage_score": 0.5,
            "damage_categories": ["unknown"],
            "description": "Image analysis not available (offline mode)",
            "key_observations": [],
            "location_clues": "",
            "rescue_priority": 5,
            "recommended_action": "Manual assessment required"
        }
        
        if not self.available or self.offline_mode:
            return default_result
        
        try:
            # Load image
            path = Path(image_path)
            if not path.exists():
                logger.error(f"Image not found: {image_path}")
                return default_result
            
            image = Image.open(path)
            
            # Generate analysis
            response = self.vision_model.generate_content(
                [self.DAMAGE_ANALYSIS_PROMPT, image],
                generation_config={
                    "max_output_tokens": 1024,
                    "temperature": 0.3
                }
            )
            
            if not response.text:
                logger.warning("Empty response from Gemini Vision")
                return default_result
            
            # Parse JSON response
            result = self._parse_json_response(response.text)
            
            # Validate and normalize
            result["damage_score"] = max(0.0, min(1.0, float(result.get("damage_score", 0.5))))
            result["damage_categories"] = result.get("damage_categories", ["unknown"])
            result["description"] = result.get("description", "No description available")
            
            return result
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return default_result
    
    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON from potentially messy LLM response.
        
        Args:
            text: Raw response text that may contain JSON
        
        Returns:
            Parsed dictionary or empty dict
        """
        try:
            # Try direct parse first
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON object
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        logger.warning(f"Could not parse JSON from response: {text[:200]}")
        return {}
    
    def summarize_situation(
        self,
        search_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a situation summary from search results.
        
        Args:
            search_results: List of search result dictionaries
        
        Returns:
            Formatted situation summary
        """
        if not search_results:
            return "No situation data available."
        
        # Build context from results
        context_parts = []
        for i, result in enumerate(search_results[:10], 1):
            payload = result.get("payload", {})
            collection = result.get("collection", "unknown")
            
            if collection == "sentinel_semantic":
                text = payload.get("text", "No text")
                context_parts.append(f"[Doc {i}] {text[:300]}")
            elif collection == "sentinel_episodic":
                desc = payload.get("description", "No description")
                damage = payload.get("damage_score", 0)
                context_parts.append(f"[Image {i}] {desc} (Damage: {damage:.0%})")
            else:
                transcript = payload.get("transcript", "No transcript")
                stress = payload.get("stress_level", 0)
                context_parts.append(f"[Audio {i}] {transcript[:200]} (Stress: {stress:.0%})")
        
        context = "\n".join(context_parts)
        
        if self.available and not self.offline_mode:
            try:
                prompt = f"""Create a brief situation summary (2-3 paragraphs) from this disaster response data:

{context}

Focus on:
1. Current damage assessment
2. Priority rescue areas
3. Recommended immediate actions"""

                response = self.text_model.generate_content(
                    prompt,
                    generation_config={"max_output_tokens": 512}
                )
                
                if response.text:
                    return response.text
                    
            except Exception as e:
                logger.error(f"Summary generation failed: {e}")
        
        # Offline fallback
        return f"""**Situation Summary (Offline Mode)**

Analyzed {len(search_results)} data points from the database.

**Data Sources:**
- Documents: {sum(1 for r in search_results if r.get('collection') == 'sentinel_semantic')}
- Images: {sum(1 for r in search_results if r.get('collection') == 'sentinel_episodic')}
- Audio: {sum(1 for r in search_results if r.get('collection') == 'sentinel_audio')}

**Key Findings:**
{context[:1000]}

**Recommendation:** Enable Gemini API for enhanced AI-powered situation analysis."""
    
    def is_available(self) -> bool:
        """Check if Gemini API is available and configured."""
        return self.available and not self.offline_mode
