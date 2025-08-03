"""External service integrations for BioNeuro-Olfactory-Fusion."""

from .github.client import GitHubClient
from .notifications.email.client import EmailClient
from .notifications.slack import SlackClient
from .auth.oauth import OAuthManager
from .webhooks.handlers import WebhookManager

__all__ = [
    'GitHubClient',
    'EmailClient', 
    'SlackClient',
    'OAuthManager',
    'WebhookManager'
]