"""Open Lilli - AI-powered PowerPoint generation tool."""

__version__ = "0.1.0"

from .chart_advisor import ChartAdvisor
from .visual_proofreader import VisualProofreader, DesignIssue, DesignIssueType, ProofreadingResult
from .flow_intelligence import FlowIntelligence, TransitionSuggestion, TransitionType, FlowAnalysisResult
from .engagement_tuner import EngagementPromptTuner, EngagementMetrics, VerbAnalysis