"""Request inspector for analyzing requests and determining routing keys."""

import logging
from typing import Dict, List, Set

from app.common.models import ClaudeRequest, Message
from app.config.log import get_logger

logger = get_logger(__name__)


class RequestInspector:
    """Analyzes Claude requests to determine appropriate routing keys."""

    # Keywords that suggest different routing types
    PLANNING_KEYWORDS = {
        'plan',
        'planning',
        'strategy',
        'approach',
        'design',
        'architecture',
        'analyze',
        'analysis',
        'research',
        'investigate',
        'study',
        'evaluate',
        'complex',
        'comprehensive',
        'detailed',
        'thorough',
        'in-depth',
        'think',
        'thinking',
        'reason',
        'reasoning',
        'logic',
        'logical',
        'solve',
        'solution',
        'problem',
        'challenge',
        'difficult',
        'hard',
        'step by step',
        'systematically',
        'methodology',
        'framework',
    }

    BACKGROUND_KEYWORDS = {
        'quick',
        'simple',
        'basic',
        'easy',
        'straightforward',
        'brief',
        'summarize',
        'summary',
        'list',
        'enumerate',
        'translate',
        'convert',
        'format',
        'formatting',
        'style',
        'lint',
        'fix typo',
        'typos',
        'spell check',
        'grammar',
        'punctuation',
        'capitalize',
        'lowercase',
        'yes',
        'no',
        'true',
        'false',
        'confirm',
        'verify',
    }

    def __init__(self):
        self.stats = {'total_requests': 0, 'routing_decisions': {'default': 0, 'planning': 0, 'background': 0}}

    def determine_routing_key(self, request: ClaudeRequest) -> str:
        """Determine routing key based on request analysis.

        Args:
            request: Claude request to analyze

        Returns:
            Routing key ('default', 'planning', or 'background')
        """
        self.stats['total_requests'] += 1

        try:
            routing_key = self._analyze_request(request)
            self.stats['routing_decisions'][routing_key] += 1

            logger.debug(f"Request routed to '{routing_key}' based on content analysis")
            return routing_key

        except Exception as e:
            logger.error(f'Error analyzing request for routing: {e}', exc_info=True)
            # Fall back to default routing
            self.stats['routing_decisions']['default'] += 1
            return 'default'

    def _analyze_request(self, request: ClaudeRequest) -> str:
        """Internal method to analyze request and determine routing.

        Args:
            request: Claude request to analyze

        Returns:
            Routing key
        """
        # Extract text content from messages
        text_content = self._extract_text_content(request.messages)

        # Analyze content characteristics
        word_count = len(text_content.split()) if text_content else 0

        # Check for explicit routing hints in metadata
        if request.metadata and hasattr(request.metadata, 'routing_preference'):
            preference = getattr(request.metadata, 'routing_preference', None)
            if preference in ['default', 'planning', 'background']:
                logger.debug(f'Using explicit routing preference: {preference}')
                return preference

        # Analyze content for keywords
        planning_score = self._calculate_keyword_score(text_content, self.PLANNING_KEYWORDS)
        background_score = self._calculate_keyword_score(text_content, self.BACKGROUND_KEYWORDS)

        # Check for complexity indicators
        complexity_score = self._calculate_complexity_score(request, text_content, word_count)

        # Make routing decision based on scores and heuristics
        return self._make_routing_decision(
            planning_score=planning_score,
            background_score=background_score,
            complexity_score=complexity_score,
            word_count=word_count,
            has_tools=bool(request.tools),
            has_thinking=bool(request.thinking),
        )

    def _extract_text_content(self, messages: List[Message]) -> str:
        """Extract all text content from messages.

        Args:
            messages: List of messages to extract text from

        Returns:
            Combined text content
        """
        text_parts = []

        for message in messages:
            if isinstance(message.content, str):
                text_parts.append(message.content)
            elif isinstance(message.content, list):
                for content_block in message.content:
                    if hasattr(content_block, 'type'):
                        if content_block.type == 'text' and hasattr(content_block, 'text'):
                            text_parts.append(content_block.text)
                        elif content_block.type == 'thinking' and hasattr(content_block, 'thinking'):
                            text_parts.append(content_block.thinking)

        return ' '.join(text_parts).lower()

    def _calculate_keyword_score(self, text: str, keywords: Set[str]) -> float:
        """Calculate score based on keyword matches.

        Args:
            text: Text to search for keywords
            keywords: Set of keywords to search for

        Returns:
            Score based on keyword matches (0.0 to 1.0)
        """
        if not text:
            return 0.0

        words = text.split()
        if not words:
            return 0.0

        matches = 0
        for keyword in keywords:
            if keyword in text:
                # Give extra weight to exact word matches vs substring matches
                if f' {keyword} ' in f' {text} ':
                    matches += 2
                else:
                    matches += 1

        # Normalize score
        max_possible_score = len(keywords) * 2
        return min(1.0, matches / max_possible_score)

    def _calculate_complexity_score(self, request: ClaudeRequest, text_content: str, word_count: int) -> float:
        """Calculate complexity score based on various factors.

        Args:
            request: Original request
            text_content: Extracted text content
            word_count: Number of words in content

        Returns:
            Complexity score (0.0 to 1.0)
        """
        score = 0.0

        # Length-based complexity
        if word_count > 500:
            score += 0.3
        elif word_count > 200:
            score += 0.1

        # Tool usage indicates complexity
        if request.tools and len(request.tools) > 0:
            score += 0.2
            if len(request.tools) > 3:
                score += 0.1

        # Thinking mode indicates complex reasoning
        if request.thinking:
            score += 0.3

        # Multiple messages suggest conversation complexity
        if len(request.messages) > 3:
            score += 0.1

        # System messages suggest complex setup
        if request.system and len(request.system) > 0:
            score += 0.1

        return min(1.0, score)

    def _make_routing_decision(self, planning_score: float, background_score: float, complexity_score: float, word_count: int, has_tools: bool, has_thinking: bool) -> str:
        """Make final routing decision based on analysis scores.

        Args:
            planning_score: Score for planning keywords
            background_score: Score for background keywords
            complexity_score: Overall complexity score
            word_count: Number of words in request
            has_tools: Whether request uses tools
            has_thinking: Whether request uses thinking mode

        Returns:
            Routing key
        """
        # Strong indicators for planning
        if planning_score > 0.3 or complexity_score > 0.5 or has_thinking or (has_tools and word_count > 100):
            return 'planning'

        # Strong indicators for background processing
        if background_score > 0.2 and complexity_score < 0.2 and word_count < 50 and not has_tools and not has_thinking:
            return 'background'

        # Default for everything else
        return 'default'

    def get_stats(self) -> Dict[str, any]:
        """Get routing statistics.

        Returns:
            Dictionary with routing statistics
        """
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self.stats = {'total_requests': 0, 'routing_decisions': {'default': 0, 'planning': 0, 'background': 0}}
