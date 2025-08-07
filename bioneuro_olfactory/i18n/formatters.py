"""Localized formatting utilities for numbers, dates, and units.

This module provides locale-aware formatting for various data types
used in the neuromorphic gas detection system.
"""

import locale
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import re

logger = logging.getLogger(__name__)


class LocalizedFormatters:
    """Provides localized formatting for various data types."""
    
    def __init__(self, language_code: str = "en"):
        self.language_code = language_code
        self.locale_map = {
            "en": "en_US.UTF-8",
            "es": "es_ES.UTF-8", 
            "fr": "fr_FR.UTF-8",
            "de": "de_DE.UTF-8",
            "zh": "zh_CN.UTF-8",
            "ja": "ja_JP.UTF-8",
            "ar": "ar_SA.UTF-8",
            "hi": "hi_IN.UTF-8",
            "pt": "pt_PT.UTF-8",
            "ru": "ru_RU.UTF-8"
        }
        
        # Number formatting preferences by locale
        self.number_formats = {
            "en": {"decimal": ".", "thousand": ",", "precision": 2},
            "es": {"decimal": ",", "thousand": ".", "precision": 2},
            "fr": {"decimal": ",", "thousand": " ", "precision": 2},
            "de": {"decimal": ",", "thousand": ".", "precision": 2},
            "zh": {"decimal": ".", "thousand": ",", "precision": 2},
            "ja": {"decimal": ".", "thousand": ",", "precision": 2},
            "ar": {"decimal": ".", "thousand": ",", "precision": 2},
            "hi": {"decimal": ".", "thousand": ",", "precision": 2},
            "pt": {"decimal": ",", "thousand": ".", "precision": 2},
            "ru": {"decimal": ",", "thousand": " ", "precision": 2}
        }
        
        # Date/time format preferences
        self.datetime_formats = {
            "en": {
                "date": "%m/%d/%Y",
                "time": "%I:%M %p", 
                "datetime": "%m/%d/%Y %I:%M %p",
                "timestamp": "%Y-%m-%d %H:%M:%S"
            },
            "es": {
                "date": "%d/%m/%Y",
                "time": "%H:%M",
                "datetime": "%d/%m/%Y %H:%M",
                "timestamp": "%Y-%m-%d %H:%M:%S"
            },
            "fr": {
                "date": "%d/%m/%Y",
                "time": "%H:%M",
                "datetime": "%d/%m/%Y %H:%M", 
                "timestamp": "%Y-%m-%d %H:%M:%S"
            },
            "de": {
                "date": "%d.%m.%Y",
                "time": "%H:%M",
                "datetime": "%d.%m.%Y %H:%M",
                "timestamp": "%Y-%m-%d %H:%M:%S"
            },
            "zh": {
                "date": "%Y/%m/%d",
                "time": "%H:%M",
                "datetime": "%Y/%m/%d %H:%M",
                "timestamp": "%Y-%m-%d %H:%M:%S"
            },
            "ja": {
                "date": "%Y/%m/%d", 
                "time": "%H:%M",
                "datetime": "%Y/%m/%d %H:%M",
                "timestamp": "%Y-%m-%d %H:%M:%S"
            },
            "ar": {
                "date": "%d/%m/%Y",
                "time": "%H:%M",
                "datetime": "%d/%m/%Y %H:%M",
                "timestamp": "%Y-%m-%d %H:%M:%S"
            },
            "hi": {
                "date": "%d/%m/%Y",
                "time": "%H:%M",
                "datetime": "%d/%m/%Y %H:%M",
                "timestamp": "%Y-%m-%d %H:%M:%S"  
            },
            "pt": {
                "date": "%d/%m/%Y",
                "time": "%H:%M",
                "datetime": "%d/%m/%Y %H:%M",
                "timestamp": "%Y-%m-%d %H:%M:%S"
            },
            "ru": {
                "date": "%d.%m.%Y",
                "time": "%H:%M",
                "datetime": "%d.%m.%Y %H:%M",
                "timestamp": "%Y-%m-%d %H:%M:%S"
            }
        }
        
        # Unit translations
        self.unit_translations = {
            "en": {
                "ppm": "ppm", "ppb": "ppb", "mg_per_m3": "mg/m³",
                "celsius": "°C", "fahrenheit": "°F", "percent": "%",
                "hz": "Hz", "ms": "ms", "seconds": "sec", "minutes": "min"
            },
            "es": {
                "ppm": "ppm", "ppb": "ppb", "mg_per_m3": "mg/m³", 
                "celsius": "°C", "fahrenheit": "°F", "percent": "%",
                "hz": "Hz", "ms": "ms", "seconds": "seg", "minutes": "min"
            },
            "fr": {
                "ppm": "ppm", "ppb": "ppb", "mg_per_m3": "mg/m³",
                "celsius": "°C", "fahrenheit": "°F", "percent": "%",
                "hz": "Hz", "ms": "ms", "seconds": "sec", "minutes": "min"
            },
            "de": {
                "ppm": "ppm", "ppb": "ppb", "mg_per_m3": "mg/m³",
                "celsius": "°C", "fahrenheit": "°F", "percent": "%", 
                "hz": "Hz", "ms": "ms", "seconds": "sek", "minutes": "min"
            },
            "zh": {
                "ppm": "ppm", "ppb": "ppb", "mg_per_m3": "mg/m³",
                "celsius": "°C", "fahrenheit": "°F", "percent": "%",
                "hz": "Hz", "ms": "ms", "seconds": "秒", "minutes": "分"
            },
            "ja": {
                "ppm": "ppm", "ppb": "ppb", "mg_per_m3": "mg/m³", 
                "celsius": "°C", "fahrenheit": "°F", "percent": "%",
                "hz": "Hz", "ms": "ms", "seconds": "秒", "minutes": "分"
            }
        }
        
        self._set_locale()
        
    def _set_locale(self):
        """Set system locale based on language."""
        locale_str = self.locale_map.get(self.language_code, "en_US.UTF-8")
        try:
            locale.setlocale(locale.LC_ALL, locale_str)
        except locale.Error:
            # Fallback to C locale
            try:
                locale.setlocale(locale.LC_ALL, "C")
                logger.warning(f"Could not set locale {locale_str}, using C locale")
            except locale.Error:
                logger.warning("Could not set any locale")
                
    def set_language(self, language_code: str):
        """Change the current language."""
        if language_code in self.locale_map:
            self.language_code = language_code
            self._set_locale()
        else:
            logger.warning(f"Unsupported language code: {language_code}")
            
    def format_number(self, value: float, precision: Optional[int] = None, unit: str = "") -> str:
        """Format number with localized separators.
        
        Args:
            value: Number to format
            precision: Decimal places (uses locale default if None)
            unit: Optional unit to append
            
        Returns:
            Formatted number string
        """
        format_config = self.number_formats.get(self.language_code, self.number_formats["en"])
        
        if precision is None:
            precision = format_config["precision"]
            
        # Format the number
        formatted = f"{value:.{precision}f}"
        
        # Replace decimal separator
        if format_config["decimal"] != ".":
            formatted = formatted.replace(".", format_config["decimal"])
            
        # Add thousand separators for large numbers
        if abs(value) >= 1000:
            parts = formatted.split(format_config["decimal"])
            integer_part = parts[0]
            
            # Insert thousand separators
            if len(integer_part) > 3:
                # Handle negative numbers
                negative = integer_part.startswith("-")
                if negative:
                    integer_part = integer_part[1:]
                    
                # Add separators from right to left
                formatted_integer = ""
                for i, digit in enumerate(reversed(integer_part)):
                    if i > 0 and i % 3 == 0:
                        formatted_integer = format_config["thousand"] + formatted_integer
                    formatted_integer = digit + formatted_integer
                    
                if negative:
                    formatted_integer = "-" + formatted_integer
                    
                # Reconstruct number
                if len(parts) > 1:
                    formatted = formatted_integer + format_config["decimal"] + parts[1]
                else:
                    formatted = formatted_integer
                    
        # Add unit if provided
        if unit:
            translated_unit = self.get_translated_unit(unit)
            formatted += " " + translated_unit
            
        return formatted
        
    def format_concentration(self, value: float, unit: str = "ppm", precision: int = 2) -> str:
        """Format concentration value with appropriate unit.
        
        Args:
            value: Concentration value
            unit: Unit (ppm, ppb, mg_per_m3)
            precision: Decimal places
            
        Returns:
            Formatted concentration string
        """
        return self.format_number(value, precision, unit)
        
    def format_timestamp(self, timestamp: float, format_type: str = "datetime", timezone_offset: Optional[int] = None) -> str:
        """Format timestamp with localized format.
        
        Args:
            timestamp: Unix timestamp
            format_type: Format type (date, time, datetime, timestamp)
            timezone_offset: Timezone offset in hours from UTC
            
        Returns:
            Formatted timestamp string
        """
        formats = self.datetime_formats.get(self.language_code, self.datetime_formats["en"])
        format_string = formats.get(format_type, formats["datetime"])
        
        # Convert timestamp to datetime
        dt = datetime.fromtimestamp(timestamp)
        
        # Apply timezone offset if provided
        if timezone_offset is not None:
            dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone(timezone_offset * 3600))
            
        return dt.strftime(format_string)
        
    def get_translated_unit(self, unit: str) -> str:
        """Get translated unit string.
        
        Args:
            unit: Unit key
            
        Returns:
            Translated unit string
        """
        translations = self.unit_translations.get(self.language_code, self.unit_translations["en"])
        return translations.get(unit, unit)
        
    def format_percentage(self, value: float, precision: int = 1) -> str:
        """Format percentage value.
        
        Args:
            value: Percentage value (0-100)
            precision: Decimal places
            
        Returns:
            Formatted percentage string
        """
        return self.format_number(value, precision, "percent")
        
    def format_temperature(self, value: float, unit: str = "celsius", precision: int = 1) -> str:
        """Format temperature value.
        
        Args:
            value: Temperature value
            unit: Temperature unit (celsius, fahrenheit)  
            precision: Decimal places
            
        Returns:
            Formatted temperature string
        """
        return self.format_number(value, precision, unit)
        
    def format_frequency(self, value: float, precision: int = 2) -> str:
        """Format frequency value in Hz.
        
        Args:
            value: Frequency value
            precision: Decimal places
            
        Returns:
            Formatted frequency string
        """
        return self.format_number(value, precision, "hz")
        
    def format_duration(self, seconds: float) -> str:
        """Format duration in appropriate units.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        if seconds < 60:
            return self.format_number(seconds, 1, "seconds")
        elif seconds < 3600:
            minutes = seconds / 60
            return self.format_number(minutes, 1, "minutes")
        else:
            hours = seconds / 3600
            return self.format_number(hours, 1, "hours")
            
    def parse_number(self, text: str) -> Optional[float]:
        """Parse localized number string back to float.
        
        Args:
            text: Formatted number string
            
        Returns:
            Parsed number or None if invalid
        """
        format_config = self.number_formats.get(self.language_code, self.number_formats["en"])
        
        try:
            # Remove unit suffix if present
            number_text = re.sub(r'\s*[a-zA-Z°%/³]+\s*$', '', text.strip())
            
            # Replace localized separators with standard format
            if format_config["thousand"]:
                number_text = number_text.replace(format_config["thousand"], "")
            if format_config["decimal"] != ".":
                number_text = number_text.replace(format_config["decimal"], ".")
                
            return float(number_text)
            
        except (ValueError, TypeError):
            logger.warning(f"Could not parse number: {text}")
            return None
            
    def get_format_info(self) -> Dict[str, Any]:
        """Get current formatting configuration.
        
        Returns:
            Dictionary with format configuration
        """
        return {
            "language_code": self.language_code,
            "number_format": self.number_formats.get(self.language_code, {}),
            "datetime_formats": self.datetime_formats.get(self.language_code, {}),
            "supported_units": list(self.unit_translations.get(self.language_code, {}).keys())
        }


# Global formatter instance
default_formatters = LocalizedFormatters()

def format_concentration(value: float, unit: str = "ppm", precision: int = 2) -> str:
    """Global function to format concentration."""
    return default_formatters.format_concentration(value, unit, precision)

def format_timestamp(timestamp: float, format_type: str = "datetime") -> str:
    """Global function to format timestamp."""
    return default_formatters.format_timestamp(timestamp, format_type)

def format_number(value: float, precision: Optional[int] = None, unit: str = "") -> str:
    """Global function to format number."""
    return default_formatters.format_number(value, precision, unit)