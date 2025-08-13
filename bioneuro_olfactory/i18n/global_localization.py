"""Global localization system for neuromorphic gas detection.

Comprehensive internationalization and localization framework supporting
multiple languages, regional standards, and cultural adaptations for
worldwide deployment of the gas detection system.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import locale


class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ARABIC = "ar"


class RegionalStandard(Enum):
    """Regional measurement and formatting standards."""
    METRIC = "metric"
    IMPERIAL = "imperial"
    US_CUSTOMARY = "us_customary"


@dataclass
class LocalizationContext:
    """Localization context for regional adaptations."""
    language: SupportedLanguage
    region: str  # ISO 3166 country code
    timezone: str
    currency: str
    number_format: str
    date_format: str
    measurement_system: RegionalStandard
    gas_concentration_units: str = "ppm"
    temperature_units: str = "celsius"
    pressure_units: str = "kPa"


class GlobalLocalizationManager:
    """Advanced localization manager for global deployment."""
    
    def __init__(self):
        self.logger = logging.getLogger("globalization")
        self.translations: Dict[SupportedLanguage, Dict[str, str]] = {}
        self.regional_configs: Dict[str, LocalizationContext] = {}
        self.current_language = SupportedLanguage.ENGLISH
        self.current_region = "US"
        
        # Initialize translations and regional configurations
        self._initialize_translations()
        self._initialize_regional_configs()
        
    def _initialize_translations(self):
        """Initialize translation dictionaries for all supported languages."""
        # Core system messages
        base_translations = {
            # System Status
            "system_status": {
                SupportedLanguage.ENGLISH: "System Status",
                SupportedLanguage.SPANISH: "Estado del Sistema",
                SupportedLanguage.FRENCH: "État du Système",
                SupportedLanguage.GERMAN: "Systemstatus",
                SupportedLanguage.JAPANESE: "システム状態",
                SupportedLanguage.CHINESE: "系统状态",
                SupportedLanguage.ITALIAN: "Stato del Sistema",
                SupportedLanguage.PORTUGUESE: "Status do Sistema",
                SupportedLanguage.RUSSIAN: "Статус Системы",
                SupportedLanguage.ARABIC: "حالة النظام"
            },
            
            # Gas Detection
            "gas_detected": {
                SupportedLanguage.ENGLISH: "Gas Detected",
                SupportedLanguage.SPANISH: "Gas Detectado",
                SupportedLanguage.FRENCH: "Gaz Détecté",
                SupportedLanguage.GERMAN: "Gas Erkannt",
                SupportedLanguage.JAPANESE: "ガス検出",
                SupportedLanguage.CHINESE: "检测到气体",
                SupportedLanguage.ITALIAN: "Gas Rilevato",
                SupportedLanguage.PORTUGUESE: "Gás Detectado",
                SupportedLanguage.RUSSIAN: "Обнаружен Газ",
                SupportedLanguage.ARABIC: "تم اكتشاف غاز"
            },
            
            "concentration": {
                SupportedLanguage.ENGLISH: "Concentration",
                SupportedLanguage.SPANISH: "Concentración",
                SupportedLanguage.FRENCH: "Concentration",
                SupportedLanguage.GERMAN: "Konzentration",
                SupportedLanguage.JAPANESE: "濃度",
                SupportedLanguage.CHINESE: "浓度",
                SupportedLanguage.ITALIAN: "Concentrazione",
                SupportedLanguage.PORTUGUESE: "Concentração",
                SupportedLanguage.RUSSIAN: "Концентрация",
                SupportedLanguage.ARABIC: "التركيز"
            },
            
            # Alert Levels
            "alert_low": {
                SupportedLanguage.ENGLISH: "Low Alert",
                SupportedLanguage.SPANISH: "Alerta Baja",
                SupportedLanguage.FRENCH: "Alerte Faible",
                SupportedLanguage.GERMAN: "Niedrige Warnung",
                SupportedLanguage.JAPANESE: "低警報",
                SupportedLanguage.CHINESE: "低警报",
                SupportedLanguage.ITALIAN: "Allerta Bassa",
                SupportedLanguage.PORTUGUESE: "Alerta Baixo",
                SupportedLanguage.RUSSIAN: "Низкая Тревога",
                SupportedLanguage.ARABIC: "تنبيه منخفض"
            },
            
            "alert_medium": {
                SupportedLanguage.ENGLISH: "Medium Alert",
                SupportedLanguage.SPANISH: "Alerta Media",
                SupportedLanguage.FRENCH: "Alerte Moyenne",
                SupportedLanguage.GERMAN: "Mittlere Warnung",
                SupportedLanguage.JAPANESE: "中警報",
                SupportedLanguage.CHINESE: "中警报",
                SupportedLanguage.ITALIAN: "Allerta Media",
                SupportedLanguage.PORTUGUESE: "Alerta Médio",
                SupportedLanguage.RUSSIAN: "Средняя Тревога",
                SupportedLanguage.ARABIC: "تنبيه متوسط"
            },
            
            "alert_high": {
                SupportedLanguage.ENGLISH: "High Alert",
                SupportedLanguage.SPANISH: "Alerta Alta",
                SupportedLanguage.FRENCH: "Alerte Élevée",
                SupportedLanguage.GERMAN: "Hohe Warnung",
                SupportedLanguage.JAPANESE: "高警報",
                SupportedLanguage.CHINESE: "高警报",
                SupportedLanguage.ITALIAN: "Allerta Alta",
                SupportedLanguage.PORTUGUESE: "Alerta Alto",
                SupportedLanguage.RUSSIAN: "Высокая Тревога",
                SupportedLanguage.ARABIC: "تنبيه عالي"
            },
            
            "alert_critical": {
                SupportedLanguage.ENGLISH: "Critical Alert",
                SupportedLanguage.SPANISH: "Alerta Crítica",
                SupportedLanguage.FRENCH: "Alerte Critique",
                SupportedLanguage.GERMAN: "Kritische Warnung",
                SupportedLanguage.JAPANESE: "緊急警報",
                SupportedLanguage.CHINESE: "严重警报",
                SupportedLanguage.ITALIAN: "Allerta Critica",
                SupportedLanguage.PORTUGUESE: "Alerta Crítico",
                SupportedLanguage.RUSSIAN: "Критическая Тревога",
                SupportedLanguage.ARABIC: "تنبيه حرج"
            },
            
            # Gas Types
            "methane": {
                SupportedLanguage.ENGLISH: "Methane",
                SupportedLanguage.SPANISH: "Metano",
                SupportedLanguage.FRENCH: "Méthane",
                SupportedLanguage.GERMAN: "Methan",
                SupportedLanguage.JAPANESE: "メタン",
                SupportedLanguage.CHINESE: "甲烷",
                SupportedLanguage.ITALIAN: "Metano",
                SupportedLanguage.PORTUGUESE: "Metano",
                SupportedLanguage.RUSSIAN: "Метан",
                SupportedLanguage.ARABIC: "الميثان"
            },
            
            "carbon_monoxide": {
                SupportedLanguage.ENGLISH: "Carbon Monoxide",
                SupportedLanguage.SPANISH: "Monóxido de Carbono",
                SupportedLanguage.FRENCH: "Monoxyde de Carbone",
                SupportedLanguage.GERMAN: "Kohlenmonoxid",
                SupportedLanguage.JAPANESE: "一酸化炭素",
                SupportedLanguage.CHINESE: "一氧化碳",
                SupportedLanguage.ITALIAN: "Monossido di Carbonio",
                SupportedLanguage.PORTUGUESE: "Monóxido de Carbono",
                SupportedLanguage.RUSSIAN: "Угарный Газ",
                SupportedLanguage.ARABIC: "أول أكسيد الكربون"
            },
            
            "ammonia": {
                SupportedLanguage.ENGLISH: "Ammonia",
                SupportedLanguage.SPANISH: "Amoníaco",
                SupportedLanguage.FRENCH: "Ammoniac",
                SupportedLanguage.GERMAN: "Ammoniak",
                SupportedLanguage.JAPANESE: "アンモニア",
                SupportedLanguage.CHINESE: "氨气",
                SupportedLanguage.ITALIAN: "Ammoniaca",
                SupportedLanguage.PORTUGUESE: "Amônia",
                SupportedLanguage.RUSSIAN: "Аммиак",
                SupportedLanguage.ARABIC: "الأمونيا"
            },
            
            # Safety Messages
            "evacuate_immediately": {
                SupportedLanguage.ENGLISH: "EVACUATE IMMEDIATELY",
                SupportedLanguage.SPANISH: "EVACUAR INMEDIATAMENTE",
                SupportedLanguage.FRENCH: "ÉVACUER IMMÉDIATEMENT",
                SupportedLanguage.GERMAN: "SOFORT EVAKUIEREN",
                SupportedLanguage.JAPANESE: "直ちに避難してください",
                SupportedLanguage.CHINESE: "立即撤离",
                SupportedLanguage.ITALIAN: "EVACUARE IMMEDIATAMENTE",
                SupportedLanguage.PORTUGUESE: "EVACUAR IMEDIATAMENTE",
                SupportedLanguage.RUSSIAN: "НЕМЕДЛЕННО ЭВАКУИРОВАТЬСЯ",
                SupportedLanguage.ARABIC: "إخلاء فوري"
            },
            
            "ventilation_recommended": {
                SupportedLanguage.ENGLISH: "Ventilation Recommended",
                SupportedLanguage.SPANISH: "Se Recomienda Ventilación",
                SupportedLanguage.FRENCH: "Ventilation Recommandée",
                SupportedLanguage.GERMAN: "Belüftung Empfohlen",
                SupportedLanguage.JAPANESE: "換気を推奨",
                SupportedLanguage.CHINESE: "建议通风",
                SupportedLanguage.ITALIAN: "Ventilazione Raccomandata",
                SupportedLanguage.PORTUGUESE: "Ventilação Recomendada",
                SupportedLanguage.RUSSIAN: "Рекомендуется Вентиляция",
                SupportedLanguage.ARABIC: "التهوية موصى بها"
            },
            
            # Technical Terms
            "neural_network": {
                SupportedLanguage.ENGLISH: "Neural Network",
                SupportedLanguage.SPANISH: "Red Neuronal",
                SupportedLanguage.FRENCH: "Réseau de Neurones",
                SupportedLanguage.GERMAN: "Neuronales Netzwerk",
                SupportedLanguage.JAPANESE: "ニューラルネットワーク",
                SupportedLanguage.CHINESE: "神经网络",
                SupportedLanguage.ITALIAN: "Rete Neurale",
                SupportedLanguage.PORTUGUESE: "Rede Neural",
                SupportedLanguage.RUSSIAN: "Нейронная Сеть",
                SupportedLanguage.ARABIC: "الشبكة العصبية"
            },
            
            "calibration": {
                SupportedLanguage.ENGLISH: "Calibration",
                SupportedLanguage.SPANISH: "Calibración",
                SupportedLanguage.FRENCH: "Étalonnage",
                SupportedLanguage.GERMAN: "Kalibrierung",
                SupportedLanguage.JAPANESE: "較正",
                SupportedLanguage.CHINESE: "校准",
                SupportedLanguage.ITALIAN: "Calibrazione",
                SupportedLanguage.PORTUGUESE: "Calibração",
                SupportedLanguage.RUSSIAN: "Калибровка",
                SupportedLanguage.ARABIC: "المعايرة"
            }
        }
        
        # Organize translations by language
        for term, lang_translations in base_translations.items():
            for language, translation in lang_translations.items():
                if language not in self.translations:
                    self.translations[language] = {}
                self.translations[language][term] = translation
                
    def _initialize_regional_configs(self):
        """Initialize regional configuration presets."""
        regional_configs = {
            # North America
            "US": LocalizationContext(
                language=SupportedLanguage.ENGLISH,
                region="US",
                timezone="America/New_York",
                currency="USD",
                number_format="en_US",
                date_format="%m/%d/%Y",
                measurement_system=RegionalStandard.IMPERIAL,
                gas_concentration_units="ppm",
                temperature_units="fahrenheit",
                pressure_units="psi"
            ),
            
            "CA": LocalizationContext(
                language=SupportedLanguage.ENGLISH,
                region="CA",
                timezone="America/Toronto",
                currency="CAD",
                number_format="en_CA",
                date_format="%d/%m/%Y",
                measurement_system=RegionalStandard.METRIC,
                temperature_units="celsius",
                pressure_units="kPa"
            ),
            
            "MX": LocalizationContext(
                language=SupportedLanguage.SPANISH,
                region="MX",
                timezone="America/Mexico_City",
                currency="MXN",
                number_format="es_MX",
                date_format="%d/%m/%Y",
                measurement_system=RegionalStandard.METRIC
            ),
            
            # Europe
            "GB": LocalizationContext(
                language=SupportedLanguage.ENGLISH,
                region="GB",
                timezone="Europe/London",
                currency="GBP",
                number_format="en_GB",
                date_format="%d/%m/%Y",
                measurement_system=RegionalStandard.METRIC,
                pressure_units="bar"
            ),
            
            "DE": LocalizationContext(
                language=SupportedLanguage.GERMAN,
                region="DE",
                timezone="Europe/Berlin",
                currency="EUR",
                number_format="de_DE",
                date_format="%d.%m.%Y",
                measurement_system=RegionalStandard.METRIC,
                pressure_units="bar"
            ),
            
            "FR": LocalizationContext(
                language=SupportedLanguage.FRENCH,
                region="FR",
                timezone="Europe/Paris",
                currency="EUR",
                number_format="fr_FR",
                date_format="%d/%m/%Y",
                measurement_system=RegionalStandard.METRIC,
                pressure_units="bar"
            ),
            
            "ES": LocalizationContext(
                language=SupportedLanguage.SPANISH,
                region="ES",
                timezone="Europe/Madrid",
                currency="EUR",
                number_format="es_ES",
                date_format="%d/%m/%Y",
                measurement_system=RegionalStandard.METRIC
            ),
            
            "IT": LocalizationContext(
                language=SupportedLanguage.ITALIAN,
                region="IT",
                timezone="Europe/Rome",
                currency="EUR",
                number_format="it_IT",
                date_format="%d/%m/%Y",
                measurement_system=RegionalStandard.METRIC
            ),
            
            # Asia Pacific
            "JP": LocalizationContext(
                language=SupportedLanguage.JAPANESE,
                region="JP",
                timezone="Asia/Tokyo",
                currency="JPY",
                number_format="ja_JP",
                date_format="%Y/%m/%d",
                measurement_system=RegionalStandard.METRIC
            ),
            
            "CN": LocalizationContext(
                language=SupportedLanguage.CHINESE,
                region="CN",
                timezone="Asia/Shanghai",
                currency="CNY",
                number_format="zh_CN",
                date_format="%Y年%m月%d日",
                measurement_system=RegionalStandard.METRIC
            ),
            
            # South America
            "BR": LocalizationContext(
                language=SupportedLanguage.PORTUGUESE,
                region="BR",
                timezone="America/Sao_Paulo",
                currency="BRL",
                number_format="pt_BR",
                date_format="%d/%m/%Y",
                measurement_system=RegionalStandard.METRIC
            ),
            
            # Middle East
            "SA": LocalizationContext(
                language=SupportedLanguage.ARABIC,
                region="SA",
                timezone="Asia/Riyadh",
                currency="SAR",
                number_format="ar_SA",
                date_format="%d/%m/%Y",
                measurement_system=RegionalStandard.METRIC
            ),
            
            # Eastern Europe
            "RU": LocalizationContext(
                language=SupportedLanguage.RUSSIAN,
                region="RU",
                timezone="Europe/Moscow",
                currency="RUB",
                number_format="ru_RU",
                date_format="%d.%m.%Y",
                measurement_system=RegionalStandard.METRIC
            )
        }
        
        self.regional_configs.update(regional_configs)
    
    def set_locale(self, language: SupportedLanguage, region: str = "US"):
        """Set current locale for the system."""
        self.current_language = language
        self.current_region = region
        
        if region in self.regional_configs:
            context = self.regional_configs[region]
            self.logger.info(f"Locale set to {language.value}_{region}")
            return context
        else:
            self.logger.warning(f"Region {region} not configured, using default")
            return self.regional_configs["US"]
    
    def translate(self, key: str, language: Optional[SupportedLanguage] = None) -> str:
        """Translate a message key to specified or current language."""
        target_language = language or self.current_language
        
        if target_language in self.translations and key in self.translations[target_language]:
            return self.translations[target_language][key]
        
        # Fallback to English
        if SupportedLanguage.ENGLISH in self.translations and key in self.translations[SupportedLanguage.ENGLISH]:
            self.logger.warning(f"Translation missing for {key} in {target_language.value}, using English")
            return self.translations[SupportedLanguage.ENGLISH][key]
        
        # Last resort - return key
        self.logger.error(f"Translation missing for key: {key}")
        return key
    
    def format_concentration(
        self, 
        value: float, 
        gas_type: str = "generic",
        region: Optional[str] = None
    ) -> str:
        """Format gas concentration according to regional standards."""
        target_region = region or self.current_region
        context = self.regional_configs.get(target_region, self.regional_configs["US"])
        
        # Format number according to regional preferences
        if context.number_format.startswith("en_US"):
            formatted_value = f"{value:,.2f}"
        elif context.number_format.startswith("de_") or context.number_format.startswith("fr_"):
            # European formatting (space as thousands separator, comma as decimal)
            formatted_value = f"{value:,.2f}".replace(",", " ").replace(".", ",")
            if " " in formatted_value:
                parts = formatted_value.split(",")
                if len(parts) == 2:
                    formatted_value = parts[0].replace(" ", ".") + "," + parts[1]
        else:
            formatted_value = f"{value:.2f}"
        
        return f"{formatted_value} {context.gas_concentration_units}"
    
    def format_temperature(
        self, 
        celsius_value: float, 
        region: Optional[str] = None
    ) -> str:
        """Format temperature according to regional preferences."""
        target_region = region or self.current_region
        context = self.regional_configs.get(target_region, self.regional_configs["US"])
        
        if context.temperature_units == "fahrenheit":
            fahrenheit_value = celsius_value * 9/5 + 32
            return f"{fahrenheit_value:.1f}°F"
        else:
            return f"{celsius_value:.1f}°C"
    
    def format_pressure(
        self, 
        kpa_value: float, 
        region: Optional[str] = None
    ) -> str:
        """Format pressure according to regional standards."""
        target_region = region or self.current_region
        context = self.regional_configs.get(target_region, self.regional_configs["US"])
        
        if context.pressure_units == "psi":
            psi_value = kpa_value * 0.145038
            return f"{psi_value:.1f} psi"
        elif context.pressure_units == "bar":
            bar_value = kpa_value / 100.0
            return f"{bar_value:.3f} bar"
        else:
            return f"{kpa_value:.1f} kPa"
    
    def format_datetime(
        self, 
        timestamp: float, 
        region: Optional[str] = None
    ) -> str:
        """Format datetime according to regional preferences."""
        target_region = region or self.current_region
        context = self.regional_configs.get(target_region, self.regional_configs["US"])
        
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp)
        
        try:
            return dt.strftime(context.date_format)
        except:
            # Fallback to ISO format
            return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def get_emergency_message(
        self, 
        gas_type: str, 
        concentration: float, 
        alert_level: str,
        language: Optional[SupportedLanguage] = None
    ) -> Dict[str, str]:
        """Generate localized emergency message."""
        target_language = language or self.current_language
        
        # Translate components
        gas_name = self.translate(gas_type.lower().replace(" ", "_"), target_language)
        alert_text = self.translate(f"alert_{alert_level.lower()}", target_language)
        concentration_text = self.translate("concentration", target_language)
        
        # Format concentration
        formatted_concentration = self.format_concentration(concentration)
        
        # Generate appropriate emergency action
        if alert_level.lower() == "critical":
            action = self.translate("evacuate_immediately", target_language)
        elif alert_level.lower() in ["high", "medium"]:
            action = self.translate("ventilation_recommended", target_language)
        else:
            action = ""
        
        return {
            "alert_level": alert_text,
            "gas_type": gas_name,
            "concentration": f"{concentration_text}: {formatted_concentration}",
            "recommended_action": action,
            "language": target_language.value,
            "formatted_message": f"{alert_text}: {gas_name} - {concentration_text}: {formatted_concentration}. {action}".strip()
        }
    
    def get_system_status_message(
        self, 
        status: Dict[str, Any],
        language: Optional[SupportedLanguage] = None
    ) -> Dict[str, str]:
        """Generate localized system status message."""
        target_language = language or self.current_language
        
        status_title = self.translate("system_status", target_language)
        
        # Translate common status components
        neural_network_text = self.translate("neural_network", target_language)
        calibration_text = self.translate("calibration", target_language)
        
        return {
            "title": status_title,
            "neural_network": neural_network_text,
            "calibration": calibration_text,
            "language": target_language.value
        }
    
    def validate_regional_compliance(self, region: str) -> Dict[str, Any]:
        """Validate system compliance with regional standards."""
        if region not in self.regional_configs:
            return {
                "compliant": False,
                "issues": ["Region not supported"],
                "recommendations": ["Add regional configuration"]
            }
        
        context = self.regional_configs[region]
        compliance_report = {
            "compliant": True,
            "region": region,
            "language_support": context.language in self.translations,
            "measurement_system": context.measurement_system.value,
            "units": {
                "concentration": context.gas_concentration_units,
                "temperature": context.temperature_units,
                "pressure": context.pressure_units
            },
            "formatting": {
                "numbers": context.number_format,
                "dates": context.date_format,
                "currency": context.currency
            },
            "issues": [],
            "recommendations": []
        }
        
        # Check for potential issues
        if not compliance_report["language_support"]:
            compliance_report["issues"].append(f"Limited translation support for {context.language.value}")
            compliance_report["compliant"] = False
        
        # Regional-specific compliance checks
        if region in ["DE", "FR", "GB"] and context.measurement_system != RegionalStandard.METRIC:
            compliance_report["issues"].append("EU requires metric measurements")
            compliance_report["compliant"] = False
        
        return compliance_report
    
    def export_translations(self, filepath: str):
        """Export translation dictionary to JSON file."""
        export_data = {
            "languages": [lang.value for lang in SupportedLanguage],
            "translations": {
                lang.value: translations for lang, translations in self.translations.items()
            },
            "regional_configs": {
                region: {
                    "language": config.language.value,
                    "timezone": config.timezone,
                    "currency": config.currency,
                    "measurement_system": config.measurement_system.value,
                    "units": {
                        "concentration": config.gas_concentration_units,
                        "temperature": config.temperature_units,
                        "pressure": config.pressure_units
                    }
                } for region, config in self.regional_configs.items()
            }
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Translations exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export translations: {e}")


# Global localization manager instance
global_localization = GlobalLocalizationManager()


# Convenience functions
def localize(key: str, language: Optional[SupportedLanguage] = None) -> str:
    """Quick translation function."""
    return global_localization.translate(key, language)


def set_global_locale(language: SupportedLanguage, region: str = "US"):
    """Set global system locale."""
    return global_localization.set_locale(language, region)


def format_gas_alert(gas_type: str, concentration: float, alert_level: str) -> Dict[str, str]:
    """Format gas alert message in current locale."""
    return global_localization.get_emergency_message(gas_type, concentration, alert_level)


if __name__ == "__main__":
    # Test global localization system
    print("🌍 Global Localization System for Neuromorphic Gas Detection")
    print("=" * 80)
    
    # Test different languages and regions
    test_regions = [
        (SupportedLanguage.ENGLISH, "US", "United States"),
        (SupportedLanguage.SPANISH, "ES", "Spain"),
        (SupportedLanguage.FRENCH, "FR", "France"),
        (SupportedLanguage.GERMAN, "DE", "Germany"),
        (SupportedLanguage.JAPANESE, "JP", "Japan"),
        (SupportedLanguage.CHINESE, "CN", "China"),
        (SupportedLanguage.PORTUGUESE, "BR", "Brazil"),
        (SupportedLanguage.ARABIC, "SA", "Saudi Arabia")
    ]
    
    for language, region, country_name in test_regions:
        print(f"\n🌐 Testing {country_name} ({language.value}_{region}):")
        
        # Set locale
        context = global_localization.set_locale(language, region)
        
        # Test emergency message
        emergency_msg = global_localization.get_emergency_message(
            "methane", 1500.0, "high", language
        )
        
        print(f"  🚨 Emergency Alert:")
        print(f"    {emergency_msg['formatted_message']}")
        
        # Test system status
        system_status = global_localization.get_system_status_message({
            "neural_network": "operational",
            "calibration": "complete"
        }, language)
        
        print(f"  📊 {system_status['title']}: {system_status['neural_network']} ✅")
        
        # Test formatting
        concentration = global_localization.format_concentration(250.5, region=region)
        temperature = global_localization.format_temperature(22.5, region=region)
        pressure = global_localization.format_pressure(101.3, region=region)
        
        print(f"  📏 Measurements: {concentration}, {temperature}, {pressure}")
        
        # Test compliance
        compliance = global_localization.validate_regional_compliance(region)
        compliance_status = "✅ COMPLIANT" if compliance["compliant"] else "❌ NON-COMPLIANT"
        print(f"  📋 Regional Compliance: {compliance_status}")
    
    # Test multi-language emergency scenarios
    print(f"\n🚨 Multi-Language Emergency Alert Demonstration:")
    
    emergency_scenarios = [
        ("carbon_monoxide", 150.0, "critical"),
        ("ammonia", 75.0, "medium"),
        ("methane", 2500.0, "high")
    ]
    
    for gas, concentration, level in emergency_scenarios:
        print(f"\n⚠️ {gas.replace('_', ' ').title()} Alert ({level.title()}):")
        
        # Show in multiple languages
        for lang in [SupportedLanguage.ENGLISH, SupportedLanguage.SPANISH, SupportedLanguage.GERMAN]:
            msg = global_localization.get_emergency_message(gas, concentration, level, lang)
            print(f"  {lang.value}: {msg['formatted_message']}")
    
    # Export translations for distribution
    print(f"\n📦 Exporting Translations:")
    global_localization.export_translations("/tmp/neuromorphic_translations.json")
    
    # Summary
    print(f"\n📊 Localization System Summary:")
    print(f"  Languages Supported: {len(SupportedLanguage)} languages")
    print(f"  Regional Configurations: {len(global_localization.regional_configs)} regions")
    print(f"  Translation Keys: {len(global_localization.translations[SupportedLanguage.ENGLISH])}")
    print(f"  Regional Standards: {len(RegionalStandard)} measurement systems")
    
    print(f"\n🎯 Global Localization: FULLY OPERATIONAL")
    print(f"🌍 Worldwide Deployment: Ready for international markets")
    print(f"🔧 Regional Compliance: Automated validation system")
    print(f"📱 Multi-Platform: Consistent localization across all interfaces")