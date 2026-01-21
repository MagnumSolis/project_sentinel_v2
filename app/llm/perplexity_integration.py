"""
Project Sentinel V2 - Perplexity Integration

Wrapper for Perplexity API (OpenAI-compatible) with offline fallback support.
Handles text generation with grounded search capabilities.
"""

from openai import OpenAI
from typing import Optional, List, Dict, Any
from pathlib import Path
from PIL import Image
import logging
import json
import re

logger = logging.getLogger(__name__)


class PerplexityIntegration:
    """
    Wrapper for Perplexity API integration with fallback support.
    
    Features:
    - Text generation with RAG context
    - Web-grounded responses via Sonar models
    - Offline template-based fallback
    - Citation formatting
    
    Note: Perplexity does not support vision/image analysis directly.
    For image analysis, use offline mode or a separate vision service.
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

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "sonar-pro",
        offline_mode: bool = False
    ):
        """
        Initialize Perplexity Integration.
        
        Args:
            api_key: Perplexity API key
            model: Model name for text generation (sonar-pro, sonar, etc.)
            offline_mode: If True, skip API calls and use templates
        """
        self.api_key = api_key
        self.model_name = model
        self.offline_mode = offline_mode
        
        self.available = False
        self.client = None
        
        if not offline_mode and api_key:
            try:
                # Initialize OpenAI-compatible client for Perplexity
                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.perplexity.ai"
                )
                
                self.available = True
                logger.info(f"Perplexity API initialized with model: {model}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize Perplexity API: {e}")
                self.available = False
        else:
            logger.info("Running in offline mode (Perplexity disabled)")
            
    def set_offline_mode(self, offline: bool):
        """
        Dynamically set the offline/online mode.
        If switching to online (offline=False), attempts to initialize the client.
        """
        self.offline_mode = offline
        
        if not offline and not self.client and self.api_key:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.perplexity.ai"
                )
                self.available = True
                logger.info(f"Perplexity API initialized dynamically with model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Perplexity API dynamically: {e}")
                self.available = False
                self.offline_mode = True
        
        # If switching to offline, we don't destroy the client, just ignore it.

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
            user_message = f"""Based on the following context, answer the user's question.

CONTEXT:
{context}

USER QUESTION: {query}

Provide a comprehensive answer citing relevant sources. If the context doesn't contain enough information, acknowledge this clearly."""

            # Generate response using OpenAI-compatible API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content
            else:
                logger.warning("Empty response from Perplexity")
                return self._generate_offline_response(query, context)
                
        except Exception as e:
            logger.error(f"Perplexity generation failed: {e}")
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
2. Provide a valid `PERPLEXITY_API_KEY`
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
        
        Note: Perplexity does not support direct image analysis.
        This method returns default values. For real image analysis,
        consider using a separate vision service.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dict with damage_score, damage_categories, description, etc.
        """
        default_result = {
            "damage_score": 0.5,
            "damage_categories": ["unknown"],
            "description": "Image analysis not available (Perplexity does not support vision)",
            "key_observations": [],
            "location_clues": "",
            "rescue_priority": 5,
            "recommended_action": "Manual assessment required"
        }
        
        # Perplexity doesn't support image analysis
        logger.info("Image analysis requested but Perplexity does not support vision. Using defaults.")
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

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=512
                )
                
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content
                    
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

**Recommendation:** Enable Perplexity API for enhanced AI-powered situation analysis."""
    
    def is_available(self) -> bool:
        """Check if Perplexity API is available and configured."""
        return self.available and not self.offline_mode
