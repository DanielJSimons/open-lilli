"""Exceptions for the Open Lilli presentation generator."""

from typing import Any, Dict, List, Optional


class StyleError(Exception):
    """Raised when style validation fails in presentation assembly."""
    
    def __init__(
        self, 
        message: str, 
        slide_index: Optional[int] = None,
        shape_name: Optional[str] = None,
        violations: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize StyleError.
        
        Args:
            message: Human-readable error message
            slide_index: Index of slide where violation occurred
            shape_name: Name of shape with style violation
            violations: List of detailed violation information
        """
        super().__init__(message)
        self.slide_index = slide_index
        self.shape_name = shape_name
        self.violations = violations or []
        
    def __str__(self) -> str:
        """Return formatted error message."""
        msg = super().__str__()
        
        if self.slide_index is not None:
            msg = f"Slide {self.slide_index}: {msg}"
            
        if self.shape_name:
            msg = f"{msg} (Shape: {self.shape_name})"
            
        if self.violations:
            violation_details = []
            for violation in self.violations:
                violation_str = f"  - {violation.get('type', 'Unknown')}: {violation.get('description', 'No description')}"
                if 'expected' in violation and 'actual' in violation:
                    violation_str += f" (expected: {violation['expected']}, actual: {violation['actual']})"
                violation_details.append(violation_str)
            
            if violation_details:
                msg += "\nViolations:\n" + "\n".join(violation_details)
        
        return msg
    
    def add_violation(
        self, 
        violation_type: str, 
        description: str, 
        expected: Optional[Any] = None, 
        actual: Optional[Any] = None,
        **kwargs
    ) -> None:
        """
        Add a style violation to this error.
        
        Args:
            violation_type: Type of violation (font_name, font_size, color, etc.)
            description: Human-readable description of the violation
            expected: Expected value
            actual: Actual value found
            **kwargs: Additional violation details
        """
        violation = {
            'type': violation_type,
            'description': description
        }
        
        if expected is not None:
            violation['expected'] = expected
        if actual is not None:
            violation['actual'] = actual
            
        violation.update(kwargs)
        self.violations.append(violation)


class ValidationConfigError(Exception):
    """Raised when validation configuration is invalid."""
    pass


class TemplateStyleError(Exception):
    """Raised when template style information is missing or invalid."""
    pass