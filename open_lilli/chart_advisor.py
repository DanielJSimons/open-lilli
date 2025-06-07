import logging
from typing import Dict, Any, List
from open_lilli.models import ChartType # Assuming ChartType enum is in models

logger = logging.getLogger(__name__)

class ChartAdvisor:
    """
    Advises on the best chart type for given data and prompt using an LLM (mocked)
    and fallback heuristics.
    """

    def __init__(self, llm_service_mock=None):
        """
        Initializes the ChartAdvisor.
        Args:
            llm_service_mock: A mock object for an LLM service for testing.
        """
        self.llm_service_mock = llm_service_mock

    def _call_llm(self, chart_data_summary: str, user_prompt: str) -> Dict[str, Any]:
        """
        (Mocked) Calls an LLM to get chart suggestions.
        In a real scenario, this would involve formatting a prompt for the LLM
        and parsing its response.
        """
        if self.llm_service_mock:
            try:
                # The mock should be callable and return a dict like:
                # {"chart_type": "bar", "highlights": ["Data shows X", "Y is prominent"]}
                # It might also raise an exception to simulate failure.
                response = self.llm_service_mock.suggest(chart_data_summary, user_prompt)

                # Validate basic response structure
                if isinstance(response, dict) and "chart_type" in response and "highlights" in response:
                    # Further validation for chart_type value
                    if response["chart_type"] not in [ct.value for ct in ChartType]:
                        logger.warning(f"LLM suggested an unsupported chart type: {response['chart_type']}")
                        # Return a structure indicating a need for fallback, or raise an error
                        # For simplicity here, let's assume it might return an invalid type that heuristics should catch
                        # or the mock itself controls the good/bad response for testing.
                        # Fallback will be triggered if chart_type is not valid or if an exception occurs.
                        return {"error": "LLM suggested unsupported chart type."}
                    return response
                else:
                    logger.warning(f"LLM response has invalid structure: {response}")
                    return {"error": "LLM response has invalid structure."}

            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return {"error": f"LLM call failed: {e}"}

        # Default mock behavior if no specific mock is provided or if it fails
        logger.warning("LLM service mock not available or call failed, cannot get LLM suggestion.")
        return {"error": "LLM service mock not available or call failed."}

    def _apply_heuristics(self, chart_data: Dict[str, Any], user_prompt: str) -> Dict[str, Any]:
        """
        Applies fallback heuristics to suggest a chart type.
        """
        num_series = 0
        if "series" in chart_data and isinstance(chart_data["series"], list):
            num_series = len(chart_data["series"])
        elif "values" in chart_data or "y" in chart_data: # Single series from older format
            num_series = 1

        categories = chart_data.get("categories", chart_data.get("x", []))
        labels = chart_data.get("labels", [])

        # Time series heuristic (simple check)
        is_time_series = False
        time_keywords = ["month", "year", "quarter", "date", "time"]
        if any(kw in user_prompt.lower() for kw in time_keywords) or \
           (categories and any(isinstance(cat, str) and any(kw in cat.lower() for kw in time_keywords) for cat in categories)):
            is_time_series = True

        if "compare" in user_prompt.lower() or "comparison" in user_prompt.lower():
            if num_series > 1 and categories:
                return {"chart_type": ChartType.COLUMN.value, "highlights": ["Highlights trends by category.", "Compares multiple series."]}
            elif num_series == 1 and categories:
                 return {"chart_type": ChartType.BAR.value, "highlights": ["Compares values for categories."]}


        if is_time_series:
            if num_series > 0:
                return {"chart_type": ChartType.LINE.value, "highlights": ["Shows trends over time."]}

        if labels and num_series == 1: # Pie/Doughnut for parts of a whole
             # Check if prompt suggests proportion/distribution
            proportion_keywords = ["proportion", "share", "distribution", "percentage", "breakdown"]
            if any(kw in user_prompt.lower() for kw in proportion_keywords):
                return {"chart_type": ChartType.DOUGHNUT.value, "highlights": ["Shows parts of a whole.", "Good for percentages."]}
            else:
                return {"chart_type": ChartType.PIE.value, "highlights": ["Shows parts of a whole."]}


        if "scatter" in user_prompt.lower() or ("correlation" in user_prompt.lower() and num_series >= 2):
             return {"chart_type": ChartType.SCATTER.value, "highlights": ["Shows relationship between two variables."]}

        if num_series > 1 and categories: # Stacked contributions or multiple lines
            if "trend" in user_prompt.lower() and "total" not in user_prompt.lower() : # multiple lines better for trends
                return {"chart_type": ChartType.LINE.value, "highlights": ["Shows trends for multiple series."]}
            else: # default to stacked area or column for multiple series
                return {"chart_type": ChartType.AREA.value, "highlights": ["Shows magnitude of change over time/categories.", "Good for stacked contributions."]}


        # Default fallback
        logger.info("Applying default fallback chart type: bar.")
        return {"chart_type": ChartType.BAR.value, "highlights": ["General purpose comparison."]}

    def suggest_chart_type(self, chart_data: Dict[str, Any], user_prompt: str) -> Dict[str, Any]:
        """
        Suggests the best chart type and key highlights.
        Tries LLM first, then falls back to heuristics.

        Args:
            chart_data: The data intended for the chart.
                        Example: {"categories": ["A", "B"], "series": [{"name": "S1", "values": [10,20]}]}
                                 {"labels": ["L1", "L2"], "values": [30,70]}
            user_prompt: A text prompt from the user describing what they want to visualize.

        Returns:
            A dictionary containing:
                "suggested_chart_type": (str) e.g., "bar", "line"
                "highlights": (List[str]) Key insights or highlights.
                "source": (str) "llm" or "heuristics"
        """
        if not chart_data or not isinstance(chart_data, dict):
            logger.warning("Chart data is empty or invalid. Cannot suggest chart type.")
            # Return a default or error response
            return {
                "suggested_chart_type": ChartType.BAR.value,
                "highlights": ["Chart data was empty or invalid."],
                "source": "error"
            }

        # Create a simple summary of chart_data for the LLM (if used)
        # This is a placeholder; a more sophisticated summary might be needed.
        chart_data_summary = f"Data has keys: {list(chart_data.keys())}."
        if "categories" in chart_data:
            chart_data_summary += f" Categories: {chart_data['categories'][:3]}..."
        if "labels" in chart_data:
            chart_data_summary += f" Labels: {chart_data['labels'][:3]}..."
        if "series" in chart_data and chart_data["series"]:
            chart_data_summary += f" Series count: {len(chart_data['series'])}."
            if chart_data['series'][0].get('name'):
                 chart_data_summary += f" First series name: {chart_data['series'][0]['name']}."

        llm_suggestion = None
        if self.llm_service_mock: # Attempt LLM call only if mock is provided
            llm_response = self._call_llm(chart_data_summary, user_prompt)
            if "error" not in llm_response and llm_response.get("chart_type") in [ct.value for ct in ChartType]:
                llm_suggestion = {
                    "suggested_chart_type": llm_response["chart_type"],
                    "highlights": llm_response.get("highlights", ["LLM suggested this chart."]),
                    "source": "llm"
                }
            else:
                logger.warning(f"LLM suggestion was invalid or resulted in error: {llm_response.get('error', 'Unknown LLM issue')}. Falling back to heuristics.")
        else:
            logger.info("No LLM service mock provided. Proceeding directly to heuristics.")


        if llm_suggestion:
            logger.info(f"LLM suggested chart type: {llm_suggestion['suggested_chart_type']}")
            return llm_suggestion
        else:
            logger.info("Falling back to heuristics for chart type suggestion.")
            heuristic_suggestion = self._apply_heuristics(chart_data, user_prompt)
            return {
                "suggested_chart_type": heuristic_suggestion["chart_type"],
                "highlights": heuristic_suggestion.get("highlights", ["Heuristics suggested this chart."]),
                "source": "heuristics"
            }
