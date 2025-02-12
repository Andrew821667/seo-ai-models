from typing import Dict, Any
from common.config.advisor_config import AdvisorConfig

class SEOAdvisor:
    def __init__(self, config: AdvisorConfig):
        self.config = config
        
    def get_recommendations(self, content: Dict[str, str]) -> Dict[str, Any]:
        # Заглушка для тестирования
        return {
            "title": "Good title",
            "content": "Content looks good",
            "score": 0.85
        }
