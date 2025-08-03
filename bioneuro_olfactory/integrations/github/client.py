"""GitHub integration client for automated issue tracking and deployment updates."""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiohttp
import asyncio

logger = logging.getLogger(__name__)


class GitHubClient:
    """GitHub API client for repository management and automation."""
    
    def __init__(self, token: Optional[str] = None, repo: Optional[str] = None):
        self.token = token or os.getenv('GITHUB_TOKEN')
        self.repo = repo or os.getenv('GITHUB_REPO', 'terragonlabs/bioneuro-olfactory-fusion')
        self.base_url = 'https://api.github.com'
        
        if not self.token:
            logger.warning("GitHub token not provided - some features will be disabled")
            
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Make authenticated request to GitHub API."""
        if not self.token:
            logger.error("Cannot make GitHub API request without token")
            return None
            
        headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'BioNeuro-Olfactory-Fusion/1.0'
        }
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method, 
                    url, 
                    headers=headers, 
                    json=data,
                    params=params
                ) as response:
                    if response.status == 200 or response.status == 201:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"GitHub API error {response.status}: {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"GitHub API request failed: {e}")
            return None
            
    async def create_issue(
        self, 
        title: str, 
        body: str, 
        labels: List[str] = None,
        assignees: List[str] = None
    ) -> Optional[Dict]:
        """Create a new GitHub issue."""
        data = {
            'title': title,
            'body': body
        }
        
        if labels:
            data['labels'] = labels
            
        if assignees:
            data['assignees'] = assignees
            
        return await self._make_request(
            'POST', 
            f'repos/{self.repo}/issues',
            data=data
        )
        
    async def update_issue(
        self, 
        issue_number: int, 
        title: Optional[str] = None,
        body: Optional[str] = None,
        state: Optional[str] = None,
        labels: Optional[List[str]] = None
    ) -> Optional[Dict]:
        """Update an existing GitHub issue."""
        data = {}
        
        if title:
            data['title'] = title
        if body:
            data['body'] = body
        if state:
            data['state'] = state
        if labels:
            data['labels'] = labels
            
        return await self._make_request(
            'PATCH',
            f'repos/{self.repo}/issues/{issue_number}',
            data=data
        )
        
    async def add_comment(self, issue_number: int, body: str) -> Optional[Dict]:
        """Add comment to GitHub issue."""
        data = {'body': body}
        
        return await self._make_request(
            'POST',
            f'repos/{self.repo}/issues/{issue_number}/comments',
            data=data
        )
        
    async def get_issues(
        self, 
        state: str = 'open',
        labels: Optional[str] = None,
        limit: int = 30
    ) -> List[Dict]:
        """Get list of GitHub issues."""
        params = {
            'state': state,
            'per_page': min(limit, 100)
        }
        
        if labels:
            params['labels'] = labels
            
        result = await self._make_request(
            'GET',
            f'repos/{self.repo}/issues',
            params=params
        )
        
        return result if result else []
        
    async def create_gas_detection_alert_issue(
        self, 
        gas_type: str, 
        concentration: float,
        confidence: float,
        location: Optional[str] = None,
        experiment_id: Optional[int] = None
    ) -> Optional[Dict]:
        """Create GitHub issue for critical gas detection alert."""
        timestamp = datetime.now().isoformat()
        
        title = f"🚨 Critical Gas Alert: {gas_type.upper()} Detected"
        
        body = f"""## Gas Detection Alert
        
**Alert Level**: CRITICAL  
**Timestamp**: {timestamp}  
**Gas Type**: {gas_type}  
**Concentration**: {concentration:.1f} ppm  
**Confidence**: {confidence:.2%}  
**Location**: {location or 'Unknown'}  
**Experiment ID**: {experiment_id or 'N/A'}  

### Details
This is an automated alert generated by the BioNeuro-Olfactory-Fusion gas detection system.

### Recommended Actions
- [ ] Verify gas detection with secondary sensors
- [ ] Evaluate evacuation procedures if necessary
- [ ] Review system logs for additional context
- [ ] Update safety protocols if needed

### System Information
- Detection System: Neuromorphic SNN Fusion
- Alert Generated: {timestamp}
- Requires immediate attention

---
*This issue was automatically created by the BioNeuro-Olfactory-Fusion monitoring system.*
        """
        
        labels = ['gas-alert', 'critical', 'automated', f'gas-{gas_type}']
        
        return await self.create_issue(
            title=title,
            body=body,
            labels=labels
        )
        
    async def create_system_error_issue(
        self,
        error_type: str,
        error_message: str,
        traceback: Optional[str] = None,
        component: Optional[str] = None
    ) -> Optional[Dict]:
        """Create GitHub issue for system errors."""
        timestamp = datetime.now().isoformat()
        
        title = f"🔧 System Error: {error_type}"
        
        body = f"""## System Error Report

**Error Type**: {error_type}  
**Component**: {component or 'Unknown'}  
**Timestamp**: {timestamp}  

### Error Message
```
{error_message}
```

### Traceback
```python
{traceback or 'No traceback available'}
```

### System Context
- Timestamp: {timestamp}
- Component: {component}
- Environment: Production

### Investigation Steps
- [ ] Review system logs
- [ ] Check component health
- [ ] Verify configuration
- [ ] Test recovery procedures

---
*This issue was automatically created by the BioNeuro-Olfactory-Fusion error monitoring system.*
        """
        
        labels = ['bug', 'automated', 'system-error']
        if component:
            labels.append(f'component-{component}')
            
        return await self.create_issue(
            title=title,
            body=body,
            labels=labels
        )
        
    async def create_deployment_notification(
        self,
        version: str,
        environment: str,
        status: str,
        changes: List[str] = None
    ) -> Optional[Dict]:
        """Create GitHub issue for deployment notifications."""
        timestamp = datetime.now().isoformat()
        
        status_emoji = {
            'success': '✅',
            'failed': '❌', 
            'in_progress': '🚀'
        }.get(status, '📝')
        
        title = f"{status_emoji} Deployment {status.title()}: v{version} to {environment}"
        
        changes_text = ""
        if changes:
            changes_text = "\n### Changes\n" + "\n".join(f"- {change}" for change in changes)
            
        body = f"""## Deployment Notification

**Version**: {version}  
**Environment**: {environment}  
**Status**: {status.upper()}  
**Timestamp**: {timestamp}  

{changes_text}

### Deployment Details
- Version: {version}
- Target Environment: {environment}
- Deployment Time: {timestamp}
- Status: {status}

### Post-Deployment Actions
- [ ] Verify system health
- [ ] Run smoke tests
- [ ] Monitor error rates
- [ ] Validate core functionality

---
*This issue was automatically created by the BioNeuro-Olfactory-Fusion deployment system.*
        """
        
        labels = ['deployment', 'automated', f'env-{environment}', f'status-{status}']
        
        issue = await self.create_issue(
            title=title,
            body=body,
            labels=labels
        )
        
        # Close issue automatically if deployment successful
        if status == 'success' and issue:
            await asyncio.sleep(5)  # Brief delay
            await self.update_issue(
                issue['number'],
                state='closed'
            )
            
        return issue
        
    async def get_repository_info(self) -> Optional[Dict]:
        """Get repository information."""
        return await self._make_request('GET', f'repos/{self.repo}')
        
    async def get_latest_release(self) -> Optional[Dict]:
        """Get latest repository release."""
        return await self._make_request('GET', f'repos/{self.repo}/releases/latest')
        
    async def get_commits(self, limit: int = 10) -> List[Dict]:
        """Get recent commits."""
        params = {'per_page': min(limit, 100)}
        
        result = await self._make_request(
            'GET',
            f'repos/{self.repo}/commits',
            params=params
        )
        
        return result if result else []
        
    async def create_webhook(
        self,
        webhook_url: str,
        events: List[str] = None,
        secret: Optional[str] = None
    ) -> Optional[Dict]:
        """Create repository webhook."""
        if not events:
            events = ['push', 'pull_request', 'issues']
            
        data = {
            'name': 'web',
            'active': True,
            'events': events,
            'config': {
                'url': webhook_url,
                'content_type': 'json'
            }
        }
        
        if secret:
            data['config']['secret'] = secret
            
        return await self._make_request(
            'POST',
            f'repos/{self.repo}/hooks',
            data=data
        )
        
    async def health_check(self) -> bool:
        """Check GitHub API connectivity and authentication."""
        try:
            result = await self._make_request('GET', 'user')
            return result is not None
        except Exception as e:
            logger.error(f"GitHub health check failed: {e}")
            return False


# Utility functions for integration
async def create_critical_gas_alert(
    gas_type: str,
    concentration: float,
    confidence: float,
    **kwargs
) -> bool:
    """Create critical gas detection alert on GitHub."""
    try:
        client = GitHubClient()
        issue = await client.create_gas_detection_alert_issue(
            gas_type=gas_type,
            concentration=concentration,
            confidence=confidence,
            **kwargs
        )
        
        if issue:
            logger.info(f"Created GitHub alert issue #{issue['number']} for {gas_type} detection")
            return True
        else:
            logger.error("Failed to create GitHub alert issue")
            return False
            
    except Exception as e:
        logger.error(f"Error creating GitHub gas alert: {e}")
        return False


async def report_system_error(
    error_type: str,
    error_message: str,
    **kwargs
) -> bool:
    """Report system error to GitHub."""
    try:
        client = GitHubClient()
        issue = await client.create_system_error_issue(
            error_type=error_type,
            error_message=error_message,
            **kwargs
        )
        
        if issue:
            logger.info(f"Created GitHub error issue #{issue['number']} for {error_type}")
            return True
        else:
            logger.error("Failed to create GitHub error issue")
            return False
            
    except Exception as e:
        logger.error(f"Error creating GitHub error report: {e}")
        return False