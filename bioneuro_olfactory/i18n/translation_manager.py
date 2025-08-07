"""Translation management for multi-language support.

This module handles loading, managing, and serving translated text
for the neuromorphic gas detection system.
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
import threading

logger = logging.getLogger(__name__)


@dataclass
class TranslationConfig:
    """Translation configuration."""
    default_language: str = "en"
    fallback_language: str = "en"
    translations_dir: str = "translations"
    supported_languages: list = field(default_factory=lambda: [
        "en",  # English
        "es",  # Spanish  
        "fr",  # French
        "de",  # German
        "zh",  # Chinese
        "ja",  # Japanese
        "ar",  # Arabic
        "hi",  # Hindi
        "pt",  # Portuguese
        "ru"   # Russian
    ])


class TranslationManager:
    """Manages translations for multi-language support."""
    
    def __init__(self, config: Optional[TranslationConfig] = None):
        self.config = config or TranslationConfig()
        self.current_language = self.config.default_language
        self._translations: Dict[str, Dict[str, str]] = {}
        self._lock = threading.RLock()
        
        # Load translations
        self._load_all_translations()
        
    def _get_translations_dir(self) -> Path:
        """Get translations directory path."""
        # Look for translations relative to this file
        current_dir = Path(__file__).parent
        translations_dir = current_dir / self.config.translations_dir
        
        if not translations_dir.exists():
            # Create default translations directory
            translations_dir.mkdir(parents=True, exist_ok=True)
            self._create_default_translations(translations_dir)
            
        return translations_dir
        
    def _create_default_translations(self, translations_dir: Path):
        """Create default translation files."""
        default_translations = {
            "system": {
                "startup": "System starting up...",
                "shutdown": "System shutting down...",
                "ready": "System ready",
                "error": "System error occurred",
                "warning": "Warning",
                "info": "Information"
            },
            "sensors": {
                "calibrating": "Calibrating sensors...",
                "calibration_complete": "Sensor calibration complete",
                "reading_error": "Sensor reading error",
                "connection_lost": "Sensor connection lost",
                "reconnecting": "Reconnecting to sensor...",
                "offline": "Sensor offline",
                "online": "Sensor online"
            },
            "alerts": {
                "gas_detected": "Gas detected: {gas_type}",
                "concentration_high": "High concentration detected: {concentration} {unit}",
                "multiple_gases": "Multiple gases detected",
                "emergency_level": "EMERGENCY: Dangerous gas levels detected",
                "acknowledge_alert": "Acknowledge Alert",
                "alert_cleared": "Alert condition cleared"
            },
            "neural": {
                "processing": "Processing neural network...",
                "inference_complete": "Inference complete",
                "model_loading": "Loading neural model...",
                "spike_train_generated": "Spike train generated",
                "fusion_processing": "Multi-modal fusion processing",
                "classification_result": "Classification: {result} (confidence: {confidence}%)"
            },
            "ui": {
                "dashboard": "Dashboard",
                "sensors": "Sensors", 
                "alerts": "Alerts",
                "settings": "Settings",
                "status": "Status",
                "history": "History",
                "login": "Login",
                "logout": "Logout",
                "save": "Save",
                "cancel": "Cancel",
                "delete": "Delete",
                "edit": "Edit",
                "view": "View",
                "refresh": "Refresh",
                "export": "Export",
                "import": "Import"
            },
            "errors": {
                "connection_failed": "Connection failed",
                "invalid_input": "Invalid input provided", 
                "permission_denied": "Permission denied",
                "file_not_found": "File not found",
                "timeout": "Operation timed out",
                "unknown_error": "Unknown error occurred",
                "validation_failed": "Input validation failed"
            },
            "units": {
                "ppm": "ppm",
                "ppb": "ppb",
                "mg_per_m3": "mg/m³",
                "celsius": "°C",
                "fahrenheit": "°F",
                "percent": "%",
                "hz": "Hz",
                "ms": "ms",
                "seconds": "seconds",
                "minutes": "minutes"
            }
        }
        
        # Create English base translation
        en_file = translations_dir / "en.json"
        with open(en_file, 'w', encoding='utf-8') as f:
            json.dump(default_translations, f, indent=2, ensure_ascii=False)
            
        # Create placeholder files for other languages
        for lang in self.config.supported_languages:
            if lang != "en":
                lang_file = translations_dir / f"{lang}.json"
                if not lang_file.exists():
                    # Create basic structure with some translated content
                    translated = self._get_basic_translations(lang, default_translations)
                    with open(lang_file, 'w', encoding='utf-8') as f:
                        json.dump(translated, f, indent=2, ensure_ascii=False)
                        
    def _get_basic_translations(self, lang: str, default_translations: Dict) -> Dict:
        """Get basic translations for a language."""
        # Basic translations for common languages
        basic_translations = {
            "es": {  # Spanish
                "system": {
                    "startup": "Iniciando sistema...",
                    "shutdown": "Cerrando sistema...",
                    "ready": "Sistema listo",
                    "error": "Error del sistema",
                    "warning": "Advertencia",
                    "info": "Información"
                },
                "alerts": {
                    "gas_detected": "Gas detectado: {gas_type}",
                    "concentration_high": "Alta concentración detectada: {concentration} {unit}",
                    "emergency_level": "EMERGENCIA: Niveles peligrosos de gas detectados"
                },
                "ui": {
                    "dashboard": "Panel de Control",
                    "sensors": "Sensores",
                    "alerts": "Alertas", 
                    "settings": "Configuración",
                    "status": "Estado"
                }
            },
            "fr": {  # French
                "system": {
                    "startup": "Démarrage du système...",
                    "shutdown": "Arrêt du système...", 
                    "ready": "Système prêt",
                    "error": "Erreur système",
                    "warning": "Avertissement",
                    "info": "Information"
                },
                "alerts": {
                    "gas_detected": "Gaz détecté: {gas_type}",
                    "concentration_high": "Concentration élevée détectée: {concentration} {unit}",
                    "emergency_level": "URGENCE: Niveaux dangereux de gaz détectés"
                },
                "ui": {
                    "dashboard": "Tableau de Bord",
                    "sensors": "Capteurs",
                    "alerts": "Alertes",
                    "settings": "Paramètres", 
                    "status": "Statut"
                }
            },
            "de": {  # German
                "system": {
                    "startup": "System startet...",
                    "shutdown": "System wird heruntergefahren...",
                    "ready": "System bereit",
                    "error": "Systemfehler",
                    "warning": "Warnung",
                    "info": "Information"
                },
                "alerts": {
                    "gas_detected": "Gas erkannt: {gas_type}",
                    "concentration_high": "Hohe Konzentration erkannt: {concentration} {unit}",
                    "emergency_level": "NOTFALL: Gefährliche Gaskonzentrationen erkannt"
                },
                "ui": {
                    "dashboard": "Armaturenbrett",
                    "sensors": "Sensoren",
                    "alerts": "Warnungen",
                    "settings": "Einstellungen",
                    "status": "Status"
                }
            }
        }
        
        # Return basic translations if available, otherwise return default
        return basic_translations.get(lang, default_translations)
        
    def _load_all_translations(self):
        """Load all available translation files."""
        translations_dir = self._get_translations_dir()
        
        for lang in self.config.supported_languages:
            self._load_translation(lang, translations_dir)
            
    def _load_translation(self, language: str, translations_dir: Path):
        """Load translation file for specific language."""
        translation_file = translations_dir / f"{language}.json"
        
        if not translation_file.exists():
            logger.warning(f"Translation file not found: {translation_file}")
            return
            
        try:
            with open(translation_file, 'r', encoding='utf-8') as f:
                translations = json.load(f)
                self._translations[language] = self._flatten_dict(translations)
                logger.info(f"Loaded {len(self._translations[language])} translations for {language}")
        except Exception as e:
            logger.error(f"Failed to load translation file {translation_file}: {e}")
            
    def _flatten_dict(self, nested_dict: Dict, parent_key: str = '', separator: str = '.') -> Dict[str, str]:
        """Flatten nested dictionary for easy key lookup."""
        items = []
        for key, value in nested_dict.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, separator).items())
            else:
                items.append((new_key, str(value)))
        return dict(items)
        
    def set_language(self, language_code: str):
        """Set the current language."""
        with self._lock:
            if language_code in self.config.supported_languages:
                old_language = self.current_language
                self.current_language = language_code
                logger.info(f"Language changed from {old_language} to {language_code}")
            else:
                logger.warning(f"Unsupported language: {language_code}")
                
    def get_language(self) -> str:
        """Get current language."""
        return self.current_language
        
    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        return self.config.supported_languages.copy()
        
    def get_text(self, key: str, **kwargs) -> str:
        """Get translated text for given key.
        
        Args:
            key: Translation key (e.g., 'alerts.gas_detected')
            **kwargs: Variables for text formatting
            
        Returns:
            Translated and formatted text
        """
        with self._lock:
            # Try current language first
            text = self._get_translation(self.current_language, key)
            
            # Fall back to fallback language
            if text is None and self.current_language != self.config.fallback_language:
                text = self._get_translation(self.config.fallback_language, key)
                
            # Last resort - return key itself
            if text is None:
                logger.warning(f"Translation not found for key: {key}")
                text = key
                
            # Format with provided kwargs
            try:
                return text.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to format translation '{key}': {e}")
                return text
                
    def _get_translation(self, language: str, key: str) -> Optional[str]:
        """Get translation for specific language and key."""
        if language not in self._translations:
            return None
        return self._translations[language].get(key)
        
    def add_translation(self, language: str, key: str, text: str):
        """Add or update a translation."""
        with self._lock:
            if language not in self._translations:
                self._translations[language] = {}
            self._translations[language][key] = text
            
    def reload_translations(self):
        """Reload all translation files."""
        with self._lock:
            self._translations.clear()
            self._load_all_translations()
            logger.info("All translations reloaded")
            
    def get_translation_stats(self) -> Dict[str, Any]:
        """Get translation statistics."""
        with self._lock:
            stats = {}
            for lang, translations in self._translations.items():
                stats[lang] = {
                    'total_keys': len(translations),
                    'sample_keys': list(translations.keys())[:5]
                }
            return {
                'current_language': self.current_language,
                'supported_languages': self.config.supported_languages,
                'loaded_languages': list(self._translations.keys()),
                'language_stats': stats
            }


# Global instance
default_translation_manager = TranslationManager()

def get_text(key: str, **kwargs) -> str:
    """Global function to get translated text."""
    return default_translation_manager.get_text(key, **kwargs)

def set_language(language_code: str):
    """Global function to set language."""
    default_translation_manager.set_language(language_code)