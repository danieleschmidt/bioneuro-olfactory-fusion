"""Internationalization and localization support for neuromorphic gas detection system.

This module provides comprehensive i18n support including:
- Multi-language support for UI and messages
- Regional sensor calibration standards
- Timezone and date/time formatting
- Cultural adaptations for alerting systems
"""

from .translation_manager import TranslationManager, get_text, set_language
from .regional_standards import RegionalStandardsManager, get_regional_config
from .formatters import LocalizedFormatters, format_concentration, format_timestamp

__all__ = [
    'TranslationManager',
    'get_text', 
    'set_language',
    'RegionalStandardsManager',
    'get_regional_config',
    'LocalizedFormatters',
    'format_concentration',
    'format_timestamp'
]

# Global translation manager instance
_translation_manager = TranslationManager()
_regional_manager = RegionalStandardsManager()
_formatters = LocalizedFormatters()

# Convenience functions
def get_text(key: str, **kwargs) -> str:
    """Get localized text."""
    return _translation_manager.get_text(key, **kwargs)

def set_language(language_code: str):
    """Set active language."""
    _translation_manager.set_language(language_code)
    _formatters.set_language(language_code)

def get_regional_config(region: str = None):
    """Get regional configuration."""
    return _regional_manager.get_config(region)

def format_concentration(value: float, unit: str = "ppm", precision: int = 2) -> str:
    """Format concentration value with localization."""
    return _formatters.format_concentration(value, unit, precision)

def format_timestamp(timestamp: float, format_type: str = "datetime") -> str:
    """Format timestamp with localization."""
    return _formatters.format_timestamp(timestamp, format_type)