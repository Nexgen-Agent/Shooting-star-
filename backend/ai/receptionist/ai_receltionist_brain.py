from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
import logging
import time
import json
import random

from database.models.reception.client_session import ClientSession
from services.auto_negotiation_service import AutoNegotiationService

logger = logging.getLogger(__name__)

class AIReceptionistBrain:
    """AI Brain handling adaptive learning, tone, and negotiation"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.negotiation_service = AutoNegotiationService(db)
        
        # Personality templates for different client types
        self.personality_templates = {
            "premium": {
                "tone": "professional yet warm",
                "formality": "high",
                "response_speed": "immediate",
                "personalization": "high"
            },
            "enterprise": {
                "tone": "highly professional",
                "formality": "very high", 
                "response_speed": "fast",
                "personalization": "medium"
            },
            "standard": {
                "tone": "friendly and helpful",
                "formality": "medium",
                "response_speed": "normal",
                "personalization": "medium"
            },
            "prospect": {
                "tone": "enthusiastic and persuasive",
                "formality": "medium",
                "response_speed": "fast",
                "personalization": "low"
            }
        }

    async def analyze_message(self, message: str, session_id: str) -> Dict[str, Any]:
        """Analyze user message for intent, sentiment, and entities"""
        try:
            start_time = time.time()
            
            # Basic NLP analysis (in production, integrate with actual NLP service)
            intent = await self._detect_intent(message)
            entities = await self._extract_entities(message)
            sentiment = await self._analyze_sentiment(message)
            
            analysis_time = time.time() - start_time
            
            return {
                "intent": intent,
                "entities": entities,
                "sentiment": sentiment,
                "confidence": await self._calculate_confidence(message, intent),
                "analysis_time": analysis_time,
                "needs_escalation": await self._check_escalation_needed(intent, sentiment)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing message: {str(e)}")
            return {
                "intent": "unknown",
                "entities": {},
                "sentiment": 0.0,
                "confidence": 0.5,
                "analysis_time": 0,
                "needs_escalation": False
            }

    async def generate_response(self, user_message: str, session_context: Dict[str, Any], 
                              message_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI response based on context and analysis"""
        try:
            start_time = time.time()
            
            client_tier = session_context.get("client_tier", "standard")
            personality = self.personality_templates.get(client_tier, self.personality_templates["standard"])
            
            # Generate response based on intent
            response_data = await self._generate_intent_response(
                user_message, message_analysis, session_context, personality
            )
            
            response_time = time.time() - start_time
            
            return {
                "content": response_data["response"],
                "intent": message_analysis["intent"],
                "entities": response_data.get("entities", {}),
                "response_time": response_time,
                "tokens_used": len(response_data["response"].split()),  # Simplified token count
                "needs_escalation": response_data.get("needs_escalation", False),
                "suggested_actions": response_data.get("suggested_actions", [])
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "content": "I apologize, but I'm having trouble processing your request. Could you please rephrase or provide more details?",
                "intent": "error",
                "response_time": 0,
                "tokens_used": 0,
                "needs_escalation": False,
                "suggested_actions": []
            }

    async def handle_negotiation(self, session_id: str, service_type: str, 
                               client_budget: float, initial_quote: float) -> Dict[str, Any]:
        """Handle price negotiation with client"""
        try:
            # Get negotiation parameters based on client tier and service type
            negotiation_strategy = await self._determine_negotiation_strategy(session_id, service_type)
            
            # Use auto-negotiation service
            negotiation_result = await self.negotiation_service.negotiate_price(
                service_type=service_type,
                client_budget=client_budget,
                initial_quote=initial_quote,
                strategy=negotiation_strategy,
                client_tier=negotiation_strategy.get("client_tier", "standard")
            )
            
            # Generate negotiation response
            response_text = await self._generate_negotiation_response(
                negotiation_result, client_budget, initial_quote, negotiation_strategy
            )
            
            return {
                "negotiated_price": negotiation_result.get("final_price"),
                "response": response_text,
                "status": negotiation_result.get("status"),
                "concessions": negotiation_result.get("concessions", []),
                "profit_margin": negotiation_result.get("profit_margin")
            }
            
        except Exception as e:
            logger.error(f"Error handling negotiation: {str(e)}")
            return {
                "negotiated_price": initial_quote,
                "response": "I understand your budget concerns. The best I can offer is our standard rate for this service.",
                "status": "rejected",
                "concessions": [],
                "profit_margin": 0.3
            }

    async def adapt_communication_style(self, session_id: str, client_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt communication style based on client feedback and interactions"""
        try:
            # Analyze feedback to adjust style
            style_adjustments = await self._analyze_feedback_for_style(client_feedback)
            
            # Update session with new style
            result = await self.db.execute(
                f"UPDATE client_sessions SET conversation_style = '{style_adjustments['new_style']}' WHERE session_id = '{session_id}'"
            )
            await self.db.commit()
            
            logger.info(f"Adapted communication style for session {session_id}: {style_adjustments['new_style']}")
            return style_adjustments
            
        except Exception as e:
            logger.error(f"Error adapting communication style: {str(e)}")
            return {"new_style": "balanced", "adjustments": []}

    # ========== PRIVATE METHODS ==========

    async def _detect_intent(self, message: str) -> str:
        """Detect intent from user message (simplified implementation)"""
        message_lower = message.lower()
        
        intent_keywords = {
            "greeting": ["hello", "hi", "hey", "good morning", "good afternoon"],
            "inquiry": ["what", "how", "when", "where", "can you", "do you"],
            "pricing": ["price", "cost", "how much", "budget", "expensive"],
            "service_request": ["need", "want", "looking for", "require", "service"],
            "negotiation": ["discount", "cheaper", "lower price", "deal", "offer"],
            "complaint": ["problem", "issue", "not working", "bad", "terrible"],
            "support": ["help", "support", "assist", "guide me"]
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return intent
        
        return "general_inquiry"

    async def _extract_entities(self, message: str) -> Dict[str, Any]:
        """Extract entities from message (simplified implementation)"""
        entities = {
            "services": [],
            "budget_mentions": [],
            "timeline_mentions": [],
            "urgency_indicators": []
        }
        
        service_keywords = {
            "logo": "design",
            "website": "development", 
            "social media": "marketing",
            "ad campaign": "marketing",
            "content": "content",
            "branding": "strategy"
        }
        
        message_lower = message.lower()
        for keyword, category in service_keywords.items():
            if keyword in message_lower:
                entities["services"].append({"service": keyword, "category": category})
        
        # Simple budget extraction
        if "$" in message or "usd" in message_lower or "dollar" in message_lower:
            entities["budget_mentions"].append("budget_discussed")
            
        return entities

    async def _analyze_sentiment(self, message: str) -> float:
        """Analyze sentiment of message (simplified implementation)"""
        positive_words = ["good", "great", "excellent", "amazing", "love", "happy", "thanks", "thank you"]
        negative_words = ["bad", "terrible", "awful", "hate", "angry", "frustrated", "disappointed"]
        
        message_lower = message.lower()
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        return (positive_count - negative_count) / total

    async def _calculate_confidence(self, message: str, intent: str) -> float:
        """Calculate confidence in intent detection"""
        # Simplified confidence calculation
        intent_keywords = {
            "greeting": ["hello", "hi", "hey"],
            "pricing": ["price", "cost", "how much"],
            "service_request": ["need", "want", "looking for"]
        }
        
        keywords = intent_keywords.get(intent, [])
        message_lower = message.lower()
        
        if any(keyword in message_lower for keyword in keywords):
            return 0.9
        else:
            return 0.7

    async def _check_escalation_needed(self, intent: str, sentiment: float) -> bool:
        """Check if human escalation is needed"""
        # Escalate negative sentiment or complex issues
        if sentiment < -0.5:
            return True
        if intent in ["complaint", "complex_issue"]:
            return True
        return False

    async def _generate_intent_response(self, message: str, analysis: Dict[str, Any], 
                                      context: Dict[str, Any], personality: Dict[str, str]) -> Dict[str, Any]:
        """Generate response based on detected intent"""
        intent = analysis["intent"]
        
        response_templates = {
            "greeting": [
                "Hello! Welcome to ShootingStar. How can I assist you today?",
                "Hi there! I'm your AI receptionist. What can I help you with?",
                "Good day! I'm here to help with your marketing and branding needs."
            ],
            "pricing": [
                "I'd be happy to discuss pricing for our services. Could you tell me more about what you're looking for?",
                "Our pricing depends on the specific services you need. Let me understand your requirements better.",
                "I can provide you with a customized quote based on your project details."
            ],
            "service_request": [
                "I understand you're interested in our services. Let me gather some details to help you best.",
                "Great! I'd love to learn more about your project requirements.",
                "I can help you with that. Could you share more details about what you need?"
            ],
            "negotiation": [
                "I understand budget is important. Let me see what options we can explore.",
                "I'd be happy to discuss pricing flexibility based on your specific needs.",
                "Let me check what we can do to accommodate your budget while maintaining quality."
            ]
        }
        
        default_responses = [
            "I understand. Let me help you with that.",
            "Thanks for sharing that. How can I assist you further?",
            "I see. Let me provide you with the best possible assistance."
        ]
        
        responses = response_templates.get(intent, default_responses)
        response = random.choice(responses)
        
        return {
            "response": self._apply_personality_tone(response, personality),
            "needs_escalation": analysis.get("needs_escalation", False),
            "suggested_actions": await self._get_suggested_actions(intent, analysis.get("entities", {}))
        }

    async def _determine_negotiation_strategy(self, session_id: str, service_type: str) -> Dict[str, Any]:
        """Determine negotiation strategy based on client and service"""
        return {
            "client_tier": "standard",  # Would be fetched from session
            "service_type": service_type,
            "flexibility": 0.2,  # 20% flexibility
            "minimum_margin": 0.25,  # 25% minimum profit margin
            "max_concessions": 2,
            "strategy_type": "cooperative"  # cooperative, competitive, accommodating
        }

    async def _generate_negotiation_response(self, result: Dict[str, Any], client_budget: float, 
                                           initial_quote: float, strategy: Dict[str, Any]) -> str:
        """Generate negotiation response text"""
        final_price = result.get("final_price", initial_quote)
        
        if final_price <= client_budget:
            return f"Great news! I can offer you this service for ${final_price:.2f}, which fits within your budget."
        elif result.get("status") == "counter_offer":
            return f"I understand your budget constraints. The best I can do is ${final_price:.2f} while maintaining our quality standards."
        else:
            return f"For this level of service, our best rate is ${final_price:.2f}. This ensures we deliver the quality your brand deserves."

    async def _analyze_feedback_for_style(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feedback to adjust communication style"""
        # Simplified style adaptation
        sentiment = feedback.get("sentiment", 0)
        clarity_rating = feedback.get("clarity", 3)
        
        if sentiment > 0.7 and clarity_rating > 4:
            return {"new_style": "current", "adjustments": []}
        elif sentiment < 0:
            return {"new_style": "more_patient", "adjustments": ["slower_pace", "more_explanations"]}
        else:
            return {"new_style": "balanced", "adjustments": ["clearer_language"]}

    def _apply_personality_tone(self, response: str, personality: Dict[str, str]) -> str:
        """Apply personality tone to response (simplified)"""
        # In production, this would use more sophisticated NLP
        return response

    async def _get_suggested_actions(self, intent: str, entities: Dict[str, Any]) -> List[str]:
        """Get suggested actions based on intent and entities"""
        actions = []
        
        if intent == "service_request" and entities.get("services"):
            actions.append("create_service_request")
            actions.append("schedule_consultation")
        
        if intent == "pricing":
            actions.append("provide_pricing_guide")
            actions.append("schedule_cost_analysis")
            
        return actions