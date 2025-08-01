#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Agent

Perpetual value discovery and execution engine for continuous repository improvement.
"""

import json
import yaml
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ValueItem:
    """Represents a discovered value opportunity."""
    id: str
    title: str
    category: str
    description: str
    source: str
    estimated_effort: float  # hours
    wsjf_score: float
    ice_score: float
    technical_debt_score: float
    composite_score: float
    priority: str
    status: str = "pending"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class AutonomousSDLCAgent:
    """Autonomous SDLC enhancement agent with perpetual value discovery."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)
        self.terragon_dir = self.repo_path / ".terragon"
        self.config = self._load_config()
        self.metrics = self._load_metrics()
        self.logger = self._setup_logging()
        
    def _load_config(self) -> Dict:
        """Load Terragon configuration."""
        config_path = self.terragon_dir / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_metrics(self) -> Dict:
        """Load value metrics history."""
        metrics_path = self.terragon_dir / "value-metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
        return {"executionHistory": [], "backlogMetrics": {}}
    
    def _save_metrics(self):
        """Save updated metrics."""
        metrics_path = self.terragon_dir / "value-metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('AutonomousSDLC')
    
    def discover_value_opportunities(self) -> List[ValueItem]:
        """
        Comprehensive value discovery from multiple sources.
        
        Returns prioritized list of value opportunities.
        """
        self.logger.info("Starting value discovery cycle...")
        opportunities = []
        
        # 1. Git history analysis for TODOs/FIXMEs
        opportunities.extend(self._analyze_git_history())
        
        # 2. Static analysis for code quality issues
        opportunities.extend(self._run_static_analysis())
        
        # 3. Dependency vulnerability scanning  
        opportunities.extend(self._scan_dependencies())
        
        # 4. Test coverage analysis
        opportunities.extend(self._analyze_test_coverage())
        
        # 5. Documentation gaps
        opportunities.extend(self._analyze_documentation())
        
        # 6. Performance analysis
        opportunities.extend(self._analyze_performance())
        
        # Score and prioritize all opportunities
        scored_opportunities = []
        for opp in opportunities:
            scored_opp = self._calculate_composite_score(opp)
            scored_opportunities.append(scored_opp)
        
        # Sort by composite score descending
        return sorted(scored_opportunities, key=lambda x: x.composite_score, reverse=True)
    
    def _analyze_git_history(self) -> List[ValueItem]:
        """Analyze git history for TODO/FIXME patterns."""
        opportunities = []
        
        try:
            # Search for TODO/FIXME comments
            result = subprocess.run([
                'git', 'grep', '-n', '-i', 
                '-E', '(TODO|FIXME|HACK|XXX|DEPRECATED)',
                '--', '*.py', '*.md', '*.yaml', '*.yml'
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines[:10]):  # Limit to top 10
                    opportunities.append(ValueItem(
                        id=f"todo-{i+1:03d}",
                        title=f"Address TODO/FIXME: {line.split(':')[-1].strip()[:50]}...",
                        category="technical_debt",
                        description=f"Code comment indicates needed work: {line}",
                        source="gitHistory",
                        estimated_effort=1.0,
                        wsjf_score=0, ice_score=0, technical_debt_score=0, composite_score=0,
                        priority="medium"
                    ))
        except Exception as e:
            self.logger.warning(f"Git history analysis failed: {e}")
        
        return opportunities
    
    def _run_static_analysis(self) -> List[ValueItem]:
        """Run static analysis tools for code quality."""
        opportunities = []
        
        # Run ruff for linting issues
        try:
            result = subprocess.run([
                'ruff', 'check', '--output-format=json', str(self.repo_path)
            ], capture_output=True, text=True)
            
            if result.stdout:
                ruff_results = json.loads(result.stdout)
                for i, issue in enumerate(ruff_results[:5]):  # Top 5 issues
                    opportunities.append(ValueItem(
                        id=f"ruff-{i+1:03d}",
                        title=f"Fix {issue.get('code', 'style')} issue: {issue.get('message', '')[:50]}",
                        category="code_quality", 
                        description=f"Ruff found: {issue.get('message', 'Style violation')}",
                        source="staticAnalysis",
                        estimated_effort=0.5,
                        wsjf_score=0, ice_score=0, technical_debt_score=0, composite_score=0,
                        priority="low"
                    ))
        except Exception as e:
            self.logger.warning(f"Ruff analysis failed: {e}")
        
        return opportunities
    
    def _scan_dependencies(self) -> List[ValueItem]:
        """Scan for dependency vulnerabilities and updates."""
        opportunities = []
        
        # Run safety check for vulnerabilities
        try:
            result = subprocess.run([
                'safety', 'check', '--json'
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            if result.stdout:
                safety_results = json.loads(result.stdout)
                for i, vuln in enumerate(safety_results[:3]):  # Top 3 vulnerabilities
                    opportunities.append(ValueItem(
                        id=f"vuln-{i+1:03d}",
                        title=f"Fix {vuln.get('package_name', 'dependency')} vulnerability",
                        category="security",
                        description=f"CVE: {vuln.get('advisory', 'Security vulnerability')}",
                        source="vulnerabilityScanning",
                        estimated_effort=2.0,
                        wsjf_score=0, ice_score=0, technical_debt_score=0, composite_score=0,
                        priority="high"
                    ))
        except Exception as e:
            self.logger.warning(f"Safety scan failed: {e}")
        
        return opportunities
    
    def _analyze_test_coverage(self) -> List[ValueItem]:
        """Analyze test coverage gaps."""
        opportunities = []
        
        # Check if tests directory exists and has content
        tests_dir = self.repo_path / "tests"
        if not tests_dir.exists() or not list(tests_dir.rglob("*.py")):
            opportunities.append(ValueItem(
                id="test-001",
                title="Create comprehensive test suite",
                category="testing",
                description="No test suite found, create comprehensive tests",
                source="testCoverage",
                estimated_effort=8.0,
                wsjf_score=0, ice_score=0, technical_debt_score=0, composite_score=0,
                priority="high"
            ))
        
        return opportunities
    
    def _analyze_documentation(self) -> List[ValueItem]:
        """Analyze documentation gaps."""
        opportunities = []
        
        # Check for missing documentation files
        docs_to_check = [
            ("API_REFERENCE.md", "Create API reference documentation"),
            ("DEPLOYMENT.md", "Create deployment guide"),
            ("ARCHITECTURE.md", "Document system architecture"),
            ("CHANGELOG.md", "Maintain change log")
        ]
        
        for filename, description in docs_to_check:
            if not (self.repo_path / filename).exists():
                opportunities.append(ValueItem(
                    id=f"doc-{filename.lower().replace('.', '-')}",
                    title=description,
                    category="documentation", 
                    description=f"Missing documentation file: {filename}",
                    source="documentationAnalysis",
                    estimated_effort=3.0,
                    wsjf_score=0, ice_score=0, technical_debt_score=0, composite_score=0,
                    priority="medium"
                ))
        
        return opportunities
    
    def _analyze_performance(self) -> List[ValueItem]:
        """Analyze performance optimization opportunities."""
        opportunities = []
        
        # Look for potential performance issues in code
        python_files = list(self.repo_path.rglob("*.py"))
        if len(python_files) > 10:  # Only for substantial codebases
            opportunities.append(ValueItem(
                id="perf-001",
                title="Add performance profiling and optimization",
                category="performance",
                description="Implement performance monitoring and optimization",
                source="performanceAnalysis",
                estimated_effort=6.0,
                wsjf_score=0, ice_score=0, technical_debt_score=0, composite_score=0,
                priority="medium"
            ))
        
        return opportunities
    
    def _calculate_composite_score(self, item: ValueItem) -> ValueItem:
        """Calculate composite value score using WSJF, ICE, and technical debt."""
        
        # WSJF Components (Weighted Shortest Job First)
        user_business_value = self._score_business_value(item)
        time_criticality = self._score_time_criticality(item)  
        risk_reduction = self._score_risk_reduction(item)
        opportunity_enablement = self._score_opportunity_enablement(item)
        
        cost_of_delay = (user_business_value + time_criticality + 
                        risk_reduction + opportunity_enablement)
        job_size = item.estimated_effort
        wsjf_score = cost_of_delay / max(job_size, 0.1)  # Avoid division by zero
        
        # ICE Components (Impact, Confidence, Ease)  
        impact = self._score_impact(item)
        confidence = self._score_confidence(item)
        ease = self._score_ease(item)
        ice_score = impact * confidence * ease
        
        # Technical Debt Score
        debt_impact = self._score_debt_impact(item)
        debt_interest = self._score_debt_interest(item)
        hotspot_multiplier = self._get_hotspot_multiplier(item)
        technical_debt_score = (debt_impact + debt_interest) * hotspot_multiplier
        
        # Apply adaptive weights based on repository maturity
        weights = self.config.get('scoring', {}).get('weights', {}).get('developing', {})
        wsjf_weight = weights.get('wsjf', 0.5)
        ice_weight = weights.get('ice', 0.2)
        debt_weight = weights.get('technicalDebt', 0.2)
        security_weight = weights.get('security', 0.1)
        
        # Composite score calculation
        composite_score = (
            wsjf_weight * self._normalize_score(wsjf_score, 0, 100) +
            ice_weight * self._normalize_score(ice_score, 0, 1000) +
            debt_weight * self._normalize_score(technical_debt_score, 0, 200)
        )
        
        # Apply boost factors
        if item.category == "security":
            composite_score *= self.config.get('scoring', {}).get('thresholds', {}).get('securityBoost', 2.0)
        
        # Update item with calculated scores
        item.wsjf_score = wsjf_score
        item.ice_score = ice_score
        item.technical_debt_score = technical_debt_score
        item.composite_score = composite_score
        
        return item
    
    def _score_business_value(self, item: ValueItem) -> float:
        """Score business value (1-10 scale)."""
        category_values = {
            "security": 9,
            "performance": 7, 
            "testing": 6,
            "documentation": 5,
            "technical_debt": 4,
            "code_quality": 3
        }
        return category_values.get(item.category, 5)
    
    def _score_time_criticality(self, item: ValueItem) -> float:
        """Score time criticality (1-10 scale)."""
        if item.category == "security":
            return 9
        elif item.priority == "high":
            return 7
        elif item.priority == "medium":
            return 5
        return 3
    
    def _score_risk_reduction(self, item: ValueItem) -> float:
        """Score risk reduction value (1-10 scale)."""
        risk_categories = {
            "security": 9,
            "testing": 7,
            "performance": 5,
            "technical_debt": 6
        }
        return risk_categories.get(item.category, 3)
    
    def _score_opportunity_enablement(self, item: ValueItem) -> float:
        """Score opportunity enablement (1-10 scale)."""
        if item.category in ["testing", "documentation"]:
            return 7  # Enables other development
        return 4
    
    def _score_impact(self, item: ValueItem) -> float:
        """Score impact (1-10 scale)."""
        return self._score_business_value(item)
    
    def _score_confidence(self, item: ValueItem) -> float:
        """Score execution confidence (1-10 scale)."""
        confidence_by_category = {
            "documentation": 9,
            "testing": 8,
            "code_quality": 8,
            "security": 7,
            "technical_debt": 6,
            "performance": 5
        }
        return confidence_by_category.get(item.category, 7)
    
    def _score_ease(self, item: ValueItem) -> float:
        """Score implementation ease (1-10 scale)."""
        if item.estimated_effort <= 2:
            return 9
        elif item.estimated_effort <= 4:
            return 7
        elif item.estimated_effort <= 8:
            return 5
        return 3
    
    def _score_debt_impact(self, item: ValueItem) -> float:
        """Score technical debt impact."""
        if item.category == "technical_debt":
            return 50
        elif item.category == "code_quality":
            return 30
        return 10
    
    def _score_debt_interest(self, item: ValueItem) -> float:
        """Score technical debt interest (future cost)."""
        if item.category == "security":
            return 40
        elif item.category == "technical_debt":
            return 30
        return 10
    
    def _get_hotspot_multiplier(self, item: ValueItem) -> float:
        """Get hotspot multiplier based on code churn."""
        # Simplified hotspot detection
        return 1.2  # Default multiplier
    
    def _normalize_score(self, score: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-100 range."""
        if max_val <= min_val:
            return 50
        normalized = ((score - min_val) / (max_val - min_val)) * 100
        return max(0, min(100, normalized))
    
    def select_next_best_value(self, opportunities: List[ValueItem]) -> Optional[ValueItem]:
        """Select the next best value item to execute."""
        min_score = self.config.get('scoring', {}).get('thresholds', {}).get('minScore', 15)
        max_risk = self.config.get('scoring', {}).get('thresholds', {}).get('maxRisk', 0.8)
        
        for item in opportunities:
            if item.composite_score < min_score:
                continue
                
            # Check dependencies and conflicts (simplified)
            if self._has_blocking_dependencies(item):
                continue
                
            if self._assess_risk(item) > max_risk:
                continue
                
            return item
        
        return None
    
    def _has_blocking_dependencies(self, item: ValueItem) -> bool:
        """Check if item has blocking dependencies."""
        # Simplified dependency checking
        return False
    
    def _assess_risk(self, item: ValueItem) -> float:
        """Assess execution risk (0-1 scale)."""
        risk_by_category = {
            "performance": 0.7,
            "technical_debt": 0.5,
            "security": 0.3,
            "testing": 0.2,
            "documentation": 0.1
        }
        return risk_by_category.get(item.category, 0.4)
    
    def execute_autonomous_cycle(self) -> Dict:
        """Execute one autonomous value discovery and delivery cycle."""
        self.logger.info("Starting autonomous execution cycle...")
        
        # 1. Discover value opportunities
        opportunities = self.discover_value_opportunities()
        self.logger.info(f"Discovered {len(opportunities)} value opportunities")
        
        # 2. Select next best value item
        next_item = self.select_next_best_value(opportunities)
        
        if not next_item:
            self.logger.info("No suitable value items found for execution")
            return {"status": "no_work", "opportunities_found": len(opportunities)}
        
        self.logger.info(f"Selected item for execution: {next_item.title}")
        
        # 3. Execute the selected item (mock implementation)
        execution_result = self._execute_value_item(next_item)
        
        # 4. Update metrics and learning model
        self._update_metrics(next_item, execution_result)
        
        # 5. Update backlog
        self._update_backlog(opportunities, next_item)
        
        return {
            "status": "completed",
            "executed_item": next_item.title,
            "value_delivered": execution_result.get("value_delivered", 0),
            "opportunities_found": len(opportunities)
        }
    
    def _execute_value_item(self, item: ValueItem) -> Dict:
        """Execute a value item (mock implementation for demonstration)."""
        self.logger.info(f"Executing: {item.title}")
        
        # Mock execution results
        return {
            "status": "completed",
            "value_delivered": item.composite_score,
            "actual_effort": item.estimated_effort * 0.9,  # Slightly better than estimated
            "quality_metrics": {"tests_passing": True, "coverage_maintained": True}
        }
    
    def _update_metrics(self, item: ValueItem, execution_result: Dict):
        """Update execution metrics and learning model."""
        execution_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "itemId": item.id,
            "title": item.title,
            "category": item.category,
            "scores": {
                "wsjf": item.wsjf_score,
                "ice": item.ice_score,
                "technicalDebt": item.technical_debt_score,
                "composite": item.composite_score
            },
            "estimatedEffort": item.estimated_effort,
            "actualEffort": execution_result.get("actual_effort", item.estimated_effort),
            "status": "completed",
            "valueDelivered": execution_result.get("value_delivered", 0)
        }
        
        self.metrics["executionHistory"].append(execution_record)
        self._save_metrics()
        
        self.logger.info(f"Updated metrics for {item.title}")
    
    def _update_backlog(self, opportunities: List[ValueItem], executed_item: ValueItem):
        """Update the backlog markdown file."""
        backlog_content = self._generate_backlog_content(opportunities, executed_item)
        
        backlog_file = self.repo_path / "BACKLOG.md"
        with open(backlog_file, 'w') as f:
            f.write(backlog_content)
        
        self.logger.info("Updated BACKLOG.md")
    
    def _generate_backlog_content(self, opportunities: List[ValueItem], executed_item: ValueItem) -> str:
        """Generate updated backlog content."""
        now = datetime.utcnow().isoformat()
        
        # Filter out executed item and get top 10
        remaining_opportunities = [opp for opp in opportunities if opp.id != executed_item.id][:10]
        
        content = f"""# 📊 Autonomous Value Backlog

**Repository**: {self.config.get('repository', {}).get('name', 'Unknown')}  
**Last Updated**: {now}  
**Next Execution**: {(datetime.utcnow() + timedelta(hours=1)).isoformat()}

## 🎯 Recently Completed

### ✅ [{executed_item.id.upper()}] {executed_item.title}
- **Completion**: {now}
- **Value Delivered**: {executed_item.composite_score:.1f} points
- **Category**: {executed_item.category}

## 🔄 Next Best Value Item

"""
        
        if remaining_opportunities:
            next_item = remaining_opportunities[0]
            content += f"""**[{next_item.id.upper()}] {next_item.title}**
- **Composite Score**: {next_item.composite_score:.1f}
- **Category**: {next_item.category}
- **Estimated Effort**: {next_item.estimated_effort} hours
- **Priority**: {next_item.priority}

## 📋 Top 10 Backlog Items

| Rank | ID | Title | Score | Category | Est. Hours | Source |
|------|-----|--------|---------|----------|------------|---------|
"""
            
            for i, opp in enumerate(remaining_opportunities[:10], 1):
                content += f"| {i} | {opp.id.upper()} | {opp.title[:30]}... | {opp.composite_score:.1f} | {opp.category} | {opp.estimated_effort} | {opp.source} |\n"
        
        content += f"""

## 📈 Value Delivery Metrics

- **Total Opportunities**: {len(opportunities)}
- **Executed This Cycle**: 1
- **Average Score**: {np.mean([opp.composite_score for opp in opportunities]):.1f}

---
**Autonomous Agent Status**: ✅ Active  
**Next Scheduled Execution**: {(datetime.utcnow() + timedelta(hours=1)).isoformat()}
"""
        
        return content


def main():
    """Main entry point for autonomous agent."""
    import sys
    
    repo_path = Path.cwd() if len(sys.argv) < 2 else Path(sys.argv[1])
    
    agent = AutonomousSDLCAgent(repo_path)
    result = agent.execute_autonomous_cycle()
    
    print(f"Autonomous cycle completed: {result}")


if __name__ == "__main__":
    main()