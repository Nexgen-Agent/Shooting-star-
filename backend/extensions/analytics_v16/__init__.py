# extensions/analytics_v16/__init__.py
"""
Analytics V16 Package - Advanced analytics and insights for Shooting Star V16
"""

from .social_analyzer import social_analyzer, SocialAnalyzerV16, SocialPost, TrendAnalysis, SentimentType
from .financial_projection import financial_projection, FinancialProjectionV16, ProjectionResult, ROIAnalysis, BudgetAllocation
from .realtime_insights import realtime_insights, RealTimeInsightsV16

__all__ = [
    'social_analyzer', 'SocialAnalyzerV16', 'SocialPost', 'TrendAnalysis', 'SentimentType',
    'financial_projection', 'FinancialProjectionV16', 'ProjectionResult', 'ROIAnalysis', 'BudgetAllocation',
    'realtime_insights', 'RealTimeInsightsV16'
]