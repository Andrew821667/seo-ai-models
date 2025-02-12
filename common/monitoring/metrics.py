from typing import Dict, Any
import time

class MetricsTracker:
    """Track and monitor model metrics"""
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.start_time = time.time()
        
    def add_metric(self, name: str, value: float) -> None:
        """Add a metric to track"""
        self.metrics[name] = value
        
    def get_metric(self, name: str) -> float:
        """Get metric value"""
        return self.metrics.get(name, 0.0)
        
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all tracked metrics"""
        metrics = self.metrics.copy()
        metrics['total_time'] = time.time() - self.start_time
        return metrics
