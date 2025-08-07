"""Regional standards and compliance management.

This module handles regional variations in gas detection standards,
safety thresholds, and compliance requirements across different
geographical regions and regulatory frameworks.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class RegionalStandard(Enum):
    """Regional standards and regulations."""
    # North America
    OSHA = "osha"          # Occupational Safety and Health Administration (US)
    NIOSH = "niosh"        # National Institute for Occupational Safety and Health (US)
    CSA = "csa"            # Canadian Standards Association
    
    # Europe  
    EU_ATEX = "eu_atex"    # EU Explosive Atmospheres Directive
    HSE = "hse"            # Health and Safety Executive (UK)
    DGUV = "dguv"          # German Social Accident Insurance
    INRS = "inrs"          # French National Research and Safety Institute
    
    # Asia Pacific
    JIS = "jis"            # Japanese Industrial Standards
    GB = "gb"              # Chinese National Standards
    AS_NZS = "as_nzs"      # Australian/New Zealand Standards
    
    # International
    ISO = "iso"            # International Organization for Standardization
    IEC = "iec"            # International Electrotechnical Commission


@dataclass
class GasThreshold:
    """Gas concentration threshold definition."""
    gas_name: str
    threshold_ppm: float
    threshold_type: str  # TWA, STEL, CEIL, etc.
    unit: str = "ppm"
    description: str = ""
    
    
@dataclass  
class RegionalConfig:
    """Regional configuration for gas detection."""
    region_code: str
    region_name: str
    standards: List[RegionalStandard]
    gas_thresholds: Dict[str, List[GasThreshold]] = field(default_factory=dict)
    units_preference: Dict[str, str] = field(default_factory=dict)
    alert_preferences: Dict[str, Any] = field(default_factory=dict)
    compliance_settings: Dict[str, Any] = field(default_factory=dict)
    

class RegionalStandardsManager:
    """Manages regional standards and compliance configurations."""
    
    def __init__(self):
        self.current_region = "us"  # Default to US
        self._regional_configs: Dict[str, RegionalConfig] = {}
        self._load_regional_configs()
        
    def _load_regional_configs(self):
        """Load regional configuration data."""
        # US/North American Standards
        us_config = RegionalConfig(
            region_code="us",
            region_name="United States",
            standards=[RegionalStandard.OSHA, RegionalStandard.NIOSH],
            gas_thresholds={
                "methane": [
                    GasThreshold("methane", 1000, "TWA", "ppm", "8-hour time-weighted average"),
                    GasThreshold("methane", 5000, "STEL", "ppm", "15-minute short-term exposure limit")
                ],
                "carbon_monoxide": [
                    GasThreshold("carbon_monoxide", 50, "TWA", "ppm", "8-hour TWA (OSHA)"),
                    GasThreshold("carbon_monoxide", 35, "TWA", "ppm", "8-hour TWA (NIOSH)"),
                    GasThreshold("carbon_monoxide", 200, "CEIL", "ppm", "Ceiling limit")
                ],
                "hydrogen_sulfide": [
                    GasThreshold("hydrogen_sulfide", 20, "CEIL", "ppm", "Ceiling limit (OSHA)"),
                    GasThreshold("hydrogen_sulfide", 10, "TWA", "ppm", "8-hour TWA (NIOSH)")
                ],
                "ammonia": [
                    GasThreshold("ammonia", 50, "TWA", "ppm", "8-hour TWA"),
                    GasThreshold("ammonia", 35, "STEL", "ppm", "15-minute STEL")
                ]
            },
            units_preference={
                "concentration": "ppm",
                "temperature": "fahrenheit",
                "flow_rate": "cfm"
            },
            alert_preferences={
                "audible_alerts": True,
                "visual_alerts": True,
                "escalation_time": 300,  # 5 minutes
                "auto_acknowledgment": False
            },
            compliance_settings={
                "documentation_required": True,
                "calibration_frequency_days": 30,
                "maintenance_log_required": True,
                "incident_reporting_required": True
            }
        )
        
        # European Union Standards
        eu_config = RegionalConfig(
            region_code="eu",
            region_name="European Union",
            standards=[RegionalStandard.EU_ATEX, RegionalStandard.ISO],
            gas_thresholds={
                "methane": [
                    GasThreshold("methane", 500, "OEL", "ppm", "Occupational exposure limit"),
                    GasThreshold("methane", 1000, "STEL", "ppm", "Short-term exposure limit")
                ],
                "carbon_monoxide": [
                    GasThreshold("carbon_monoxide", 30, "OEL", "ppm", "8-hour OEL"),
                    GasThreshold("carbon_monoxide", 100, "STEL", "ppm", "15-minute STEL")
                ],
                "hydrogen_sulfide": [
                    GasThreshold("hydrogen_sulfide", 5, "OEL", "ppm", "8-hour OEL"),
                    GasThreshold("hydrogen_sulfide", 10, "STEL", "ppm", "15-minute STEL")
                ]
            },
            units_preference={
                "concentration": "mg_per_m3",
                "temperature": "celsius",
                "flow_rate": "m3_per_h"
            },
            alert_preferences={
                "audible_alerts": True,
                "visual_alerts": True,
                "escalation_time": 180,  # 3 minutes
                "auto_acknowledgment": False
            },
            compliance_settings={
                "documentation_required": True,
                "calibration_frequency_days": 90,
                "atex_certification_required": True,
                "ce_marking_required": True
            }
        )
        
        # Japanese Standards
        jp_config = RegionalConfig(
            region_code="jp",
            region_name="Japan",
            standards=[RegionalStandard.JIS],
            gas_thresholds={
                "methane": [
                    GasThreshold("methane", 1000, "ACL", "ppm", "Administrative control level")
                ],
                "carbon_monoxide": [
                    GasThreshold("carbon_monoxide", 50, "ACL", "ppm", "Administrative control level"),
                    GasThreshold("carbon_monoxide", 200, "OEL", "ppm", "Occupational exposure limit")
                ]
            },
            units_preference={
                "concentration": "ppm",
                "temperature": "celsius",
                "flow_rate": "m3_per_min"
            },
            alert_preferences={
                "audible_alerts": True,
                "visual_alerts": True,
                "escalation_time": 300
            }
        )
        
        # Chinese Standards
        cn_config = RegionalConfig(
            region_code="cn",
            region_name="China",
            standards=[RegionalStandard.GB],
            gas_thresholds={
                "methane": [
                    GasThreshold("methane", 300, "MAC", "ppm", "Maximum allowable concentration")
                ],
                "carbon_monoxide": [
                    GasThreshold("carbon_monoxide", 30, "PC-TWA", "ppm", "Permissible concentration - TWA"),
                    GasThreshold("carbon_monoxide", 60, "PC-STEL", "ppm", "Permissible concentration - STEL")
                ]
            },
            units_preference={
                "concentration": "mg_per_m3",
                "temperature": "celsius",
                "flow_rate": "m3_per_h"
            }
        )
        
        # Store configurations
        self._regional_configs["us"] = us_config
        self._regional_configs["eu"] = eu_config  
        self._regional_configs["jp"] = jp_config
        self._regional_configs["cn"] = cn_config
        
        # Add some additional regions
        self._add_derived_configs()
        
    def _add_derived_configs(self):
        """Add configurations derived from main regions."""
        # Canada (similar to US)
        ca_config = RegionalConfig(
            region_code="ca",
            region_name="Canada", 
            standards=[RegionalStandard.CSA],
            gas_thresholds=self._regional_configs["us"].gas_thresholds.copy(),
            units_preference=self._regional_configs["us"].units_preference.copy(),
            alert_preferences=self._regional_configs["us"].alert_preferences.copy()
        )
        
        # UK (EU-based but with some differences)
        uk_config = RegionalConfig(
            region_code="uk",
            region_name="United Kingdom",
            standards=[RegionalStandard.HSE],
            gas_thresholds=self._regional_configs["eu"].gas_thresholds.copy(),
            units_preference=self._regional_configs["eu"].units_preference.copy(),
            alert_preferences=self._regional_configs["eu"].alert_preferences.copy()
        )
        
        # Australia/New Zealand
        au_config = RegionalConfig(
            region_code="au",
            region_name="Australia",
            standards=[RegionalStandard.AS_NZS],
            gas_thresholds=self._regional_configs["us"].gas_thresholds.copy(),
            units_preference=self._regional_configs["us"].units_preference.copy(),
            alert_preferences=self._regional_configs["us"].alert_preferences.copy()
        )
        
        self._regional_configs["ca"] = ca_config
        self._regional_configs["uk"] = uk_config
        self._regional_configs["au"] = au_config
        
    def set_region(self, region_code: str):
        """Set the current region."""
        if region_code in self._regional_configs:
            self.current_region = region_code
            logger.info(f"Region set to: {region_code}")
        else:
            logger.warning(f"Unknown region code: {region_code}")
            
    def get_current_region(self) -> str:
        """Get current region code."""
        return self.current_region
        
    def get_supported_regions(self) -> List[str]:
        """Get list of supported region codes."""
        return list(self._regional_configs.keys())
        
    def get_config(self, region_code: Optional[str] = None) -> Optional[RegionalConfig]:
        """Get regional configuration."""
        region = region_code or self.current_region
        return self._regional_configs.get(region)
        
    def get_gas_thresholds(self, gas_name: str, region_code: Optional[str] = None) -> List[GasThreshold]:
        """Get gas thresholds for specific gas and region."""
        config = self.get_config(region_code)
        if not config:
            return []
        return config.gas_thresholds.get(gas_name, [])
        
    def get_threshold_for_type(self, gas_name: str, threshold_type: str, region_code: Optional[str] = None) -> Optional[GasThreshold]:
        """Get specific threshold for gas and type."""
        thresholds = self.get_gas_thresholds(gas_name, region_code)
        for threshold in thresholds:
            if threshold.threshold_type == threshold_type:
                return threshold
        return None
        
    def check_compliance(self, gas_name: str, concentration_ppm: float, region_code: Optional[str] = None) -> Dict[str, Any]:
        """Check if concentration complies with regional standards."""
        thresholds = self.get_gas_thresholds(gas_name, region_code)
        
        if not thresholds:
            return {
                "compliant": None,
                "message": f"No thresholds defined for {gas_name}",
                "violated_thresholds": []
            }
            
        violated_thresholds = []
        for threshold in thresholds:
            if concentration_ppm > threshold.threshold_ppm:
                violated_thresholds.append({
                    "threshold_type": threshold.threshold_type,
                    "limit_ppm": threshold.threshold_ppm,
                    "current_ppm": concentration_ppm,
                    "description": threshold.description
                })
                
        compliant = len(violated_thresholds) == 0
        
        return {
            "compliant": compliant,
            "message": "Compliant with all thresholds" if compliant else f"Violates {len(violated_thresholds)} thresholds",
            "violated_thresholds": violated_thresholds,
            "all_thresholds": [
                {
                    "threshold_type": t.threshold_type,
                    "limit_ppm": t.threshold_ppm,
                    "description": t.description
                } for t in thresholds
            ]
        }
        
    def get_preferred_units(self, measurement_type: str, region_code: Optional[str] = None) -> str:
        """Get preferred units for measurement type."""
        config = self.get_config(region_code)
        if not config:
            return "ppm"  # Default
        return config.units_preference.get(measurement_type, "ppm")
        
    def get_alert_preferences(self, region_code: Optional[str] = None) -> Dict[str, Any]:
        """Get alert preferences for region."""
        config = self.get_config(region_code)
        if not config:
            return {
                "audible_alerts": True,
                "visual_alerts": True,
                "escalation_time": 300
            }
        return config.alert_preferences
        
    def get_compliance_requirements(self, region_code: Optional[str] = None) -> Dict[str, Any]:
        """Get compliance requirements for region."""
        config = self.get_config(region_code)
        if not config:
            return {}
        return config.compliance_settings
        
    def convert_concentration(self, value_ppm: float, target_unit: str, gas_molecular_weight: float = 16.0) -> float:
        """Convert concentration from ppm to other units.
        
        Args:
            value_ppm: Concentration in ppm
            target_unit: Target unit (ppm, ppb, mg_per_m3)
            gas_molecular_weight: Molecular weight of gas (for mg/m3 conversion)
            
        Returns:
            Converted concentration value
        """
        if target_unit == "ppm":
            return value_ppm
        elif target_unit == "ppb":
            return value_ppm * 1000
        elif target_unit == "mg_per_m3":
            # Convert ppm to mg/m³ using molecular weight
            # mg/m³ = (ppm × molecular weight × 1000) / 24.45 (at 25°C, 1 atm)
            return (value_ppm * gas_molecular_weight * 1000) / 24.45
        else:
            logger.warning(f"Unknown concentration unit: {target_unit}")
            return value_ppm
            
    def get_regional_summary(self) -> Dict[str, Any]:
        """Get summary of all regional configurations."""
        summary = {}
        for region_code, config in self._regional_configs.items():
            summary[region_code] = {
                "name": config.region_name,
                "standards": [std.value for std in config.standards],
                "supported_gases": list(config.gas_thresholds.keys()),
                "preferred_units": config.units_preference,
                "compliance_features": list(config.compliance_settings.keys()) if config.compliance_settings else []
            }
        return summary


# Global instance
default_regional_manager = RegionalStandardsManager()

def get_regional_config(region_code: Optional[str] = None) -> Optional[RegionalConfig]:
    """Global function to get regional configuration.""" 
    return default_regional_manager.get_config(region_code)

def check_gas_compliance(gas_name: str, concentration_ppm: float, region_code: Optional[str] = None) -> Dict[str, Any]:
    """Global function to check gas compliance."""
    return default_regional_manager.check_compliance(gas_name, concentration_ppm, region_code)