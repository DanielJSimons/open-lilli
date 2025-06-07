import pytest
from unittest.mock import MagicMock

from open_lilli.chart_advisor import ChartAdvisor
from open_lilli.models import ChartType # Ensure ChartType is imported

# Sample chart data for tests
SAMPLE_CATEGORICAL_DATA_SINGLE_SERIES = {
    "categories": ["Q1", "Q2", "Q3", "Q4"],
    "series": [{"name": "Sales", "values": [100, 150, 120, 180]}]
}
SAMPLE_CATEGORICAL_DATA_MULTI_SERIES = {
    "categories": ["Jan", "Feb", "Mar"],
    "series": [
        {"name": "Product A", "values": [10, 20, 30]},
        {"name": "Product B", "values": [15, 25, 35]}
    ]
}
SAMPLE_LABEL_VALUE_DATA = { # Suitable for pie/doughnut
    "labels": ["Region A", "Region B", "Region C"],
    "series": [{"name": "Market Share", "values": [40, 35, 25]}] # Corrected to match NativeChartData structure
}
SAMPLE_SCATTER_DATA = {
    "series": [
        {"name": "Metric X", "values": [1, 2, 3, 4, 5]}, # Should be x-values
        {"name": "Metric Y", "values": [5, 4, 3, 2, 1]}  # Should be y-values
    ]
    # For scatter, "categories" might represent x-values if only one y-series,
    # or data might be structured as list of (x,y) points.
    # Current heuristics might need actual x,y keys or rely on prompt.
    # Let's assume a structure heuristic might look for, or rely on prompt:
    # {"x_values": [1,2,3,4,5], "y_values": [5,4,3,2,1], "type": "scatter"}
}

@pytest.fixture
def chart_advisor_no_llm():
    return ChartAdvisor(llm_service_mock=None)

@pytest.fixture
def mock_llm_service():
    return MagicMock()

@pytest.fixture
def chart_advisor_with_llm(mock_llm_service):
    return ChartAdvisor(llm_service_mock=mock_llm_service)

# Tests for LLM interaction
def test_suggest_chart_type_llm_success(chart_advisor_with_llm, mock_llm_service):
    mock_llm_service.suggest.return_value = {
        "chart_type": ChartType.LINE.value,
        "highlights": ["LLM highlight 1", "LLM highlight 2"]
    }
    result = chart_advisor_with_llm.suggest_chart_type(SAMPLE_CATEGORICAL_DATA_SINGLE_SERIES, "Show trends over time")
    assert result["suggested_chart_type"] == ChartType.LINE.value
    assert result["highlights"] == ["LLM highlight 1", "LLM highlight 2"]
    assert result["source"] == "llm"
    mock_llm_service.suggest.assert_called_once()

def test_suggest_chart_type_llm_failure_fallback_to_heuristics(chart_advisor_with_llm, mock_llm_service):
    mock_llm_service.suggest.side_effect = Exception("LLM API Error")
    # Heuristics for this data and prompt ("show trends") should suggest line
    result = chart_advisor_with_llm.suggest_chart_type(SAMPLE_CATEGORICAL_DATA_MULTI_SERIES, "show trends over time for products")
    assert result["source"] == "heuristics"
    assert result["suggested_chart_type"] == ChartType.LINE.value # Expected heuristic fallback
    mock_llm_service.suggest.assert_called_once()

def test_suggest_chart_type_llm_unsupported_type_fallback_to_heuristics(chart_advisor_with_llm, mock_llm_service):
    mock_llm_service.suggest.return_value = {
        "chart_type": "unsupported_fancy_chart", # Invalid chart type
        "highlights": ["This is too fancy"]
    }
    result = chart_advisor_with_llm.suggest_chart_type(SAMPLE_CATEGORICAL_DATA_SINGLE_SERIES, "Make it fancy")
    assert result["source"] == "heuristics"
    # Depending on data/prompt, check specific heuristic output e.g. bar for single series categorical
    assert result["suggested_chart_type"] == ChartType.BAR.value
    mock_llm_service.suggest.assert_called_once()

def test_suggest_chart_type_llm_invalid_response_structure(chart_advisor_with_llm, mock_llm_service):
    mock_llm_service.suggest.return_value = {"wrong_key": "some_chart"} # Invalid structure
    result = chart_advisor_with_llm.suggest_chart_type(SAMPLE_CATEGORICAL_DATA_SINGLE_SERIES, "User prompt")
    assert result["source"] == "heuristics"
    assert result["suggested_chart_type"] == ChartType.BAR.value # Default fallback for this data
    mock_llm_service.suggest.assert_called_once()

# Tests for heuristics (when no LLM or LLM fails)
def test_suggest_chart_type_no_llm_uses_heuristics(chart_advisor_no_llm):
    result = chart_advisor_no_llm.suggest_chart_type(SAMPLE_CATEGORICAL_DATA_SINGLE_SERIES, "Show sales per quarter")
    assert result["source"] == "heuristics"
    # Based on current heuristics (single series, categorical) -> bar
    assert result["suggested_chart_type"] == ChartType.BAR.value

# Heuristic Test Cases
@pytest.mark.parametrize("data, prompt, expected_type, description", [
    (SAMPLE_CATEGORICAL_DATA_SINGLE_SERIES, "Sales per Q", ChartType.BAR.value, "Single series categorical -> BAR"),
    (SAMPLE_CATEGORICAL_DATA_MULTI_SERIES, "Compare products", ChartType.COLUMN.value, "Multi-series categorical comparison -> COLUMN"),
    (SAMPLE_CATEGORICAL_DATA_MULTI_SERIES, "Show trends over time", ChartType.LINE.value, "Multi-series time trend -> LINE"),
    (SAMPLE_LABEL_VALUE_DATA, "Market share distribution", ChartType.DOUGHNUT.value, "Label/value for distribution -> DOUGHNUT"),
    (SAMPLE_LABEL_VALUE_DATA, "Market share breakdown", ChartType.DOUGHNUT.value, "Label/value for breakdown -> DOUGHNUT"),
    (SAMPLE_LABEL_VALUE_DATA, "Market share proportions", ChartType.DOUGHNUT.value, "Label/value for proportions -> DOUGHNUT"),
    (SAMPLE_LABEL_VALUE_DATA, "Market share general", ChartType.PIE.value, "Label/value general -> PIE"),
    ({"categories": ["2020", "2021", "2022"], "series": [{"name":"A", "values":[1,2,3]}, {"name":"B", "values":[2,3,4]}]}, "Show total magnitude over years", ChartType.AREA.value, "Time categories, multiple series, magnitude -> AREA"),
    (SAMPLE_SCATTER_DATA, "Show correlation between X and Y", ChartType.SCATTER.value, "Scatter data & prompt -> SCATTER (heuristic might need specific data keys or rely heavily on prompt)"),
    (SAMPLE_CATEGORICAL_DATA_SINGLE_SERIES, "Just show the data", ChartType.BAR.value, "Generic prompt, single series -> BAR (default like)"),
    (SAMPLE_CATEGORICAL_DATA_MULTI_SERIES, "Overview of products", ChartType.AREA.value, "Generic prompt, multi-series -> AREA (default like for multi-series)"),
])
def test_heuristics(chart_advisor_no_llm, data, prompt, expected_type, description):
    result = chart_advisor_no_llm.suggest_chart_type(data, prompt)
    assert result["suggested_chart_type"] == expected_type, f"Failed heuristic test: {description}"
    assert result["source"] == "heuristics"

def test_suggest_chart_type_empty_data(chart_advisor_no_llm):
    result = chart_advisor_no_llm.suggest_chart_type({}, "Any prompt")
    assert result["source"] == "error" # Or "heuristics" with a default if that's the behavior
    assert result["suggested_chart_type"] == ChartType.BAR.value # Default fallback
    assert "Chart data was empty or invalid" in result["highlights"][0]

def test_suggest_chart_type_invalid_data_type(chart_advisor_no_llm):
    result = chart_advisor_no_llm.suggest_chart_type("not a dict", "Any prompt")
    assert result["source"] == "error"
    assert result["suggested_chart_type"] == ChartType.BAR.value
    assert "Chart data was empty or invalid" in result["highlights"][0]
