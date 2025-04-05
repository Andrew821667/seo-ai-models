"""
Demonstration script for the SEO AI Models parsers.
Shows how to use the parsing components for different tasks.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Any, Optional

from seo_ai_models.parsers.crawlers.web_crawler import WebCrawler
from seo_ai_models.parsers.extractors.content_extractor import ContentExtractor
from seo_ai_models.parsers.extractors.meta_extractor import MetaExtractor
from seo_ai_models.parsers.analyzers.serp_analyzer import SERPAnalyzer
from seo_ai_models.parsers.parsing_pipeline import ParsingPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_results(results: Dict[str, Any], output_file: Optional[str] = None) -> None:
    """
    Save or print results.
    
    Args:
        results: Results to save
        output_file: Output filename (if None, print to stdout)
    """
    if output_file:
        directory = os.path.dirname(output_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_file}")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))

def demo_web_crawler(args):
    """Run WebCrawler demo."""
    crawler = WebCrawler(
        base_url=args.url,
        max_pages=args.max_pages,
        delay=args.delay,
        respect_robots=not args.ignore_robots
    )
    
    results = crawler.crawl()
    save_results(results, args.output)
    
def demo_content_extractor(args):
    """Run ContentExtractor demo."""
    extractor = ContentExtractor()
    
    # Read HTML from file or URL
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        results = extractor.extract_content(html_content, args.url)
    elif args.url:
        import requests
        response = requests.get(args.url)
        if response.status_code == 200:
            results = extractor.extract_content(response.text, args.url)
        else:
            results = {"error": f"Failed to fetch URL: HTTP {response.status_code}"}
    else:
        results = {"error": "No input specified"}
        
    save_results(results, args.output)

def demo_meta_extractor(args):
    """Run MetaExtractor demo."""
    extractor = MetaExtractor()
    
    # Read HTML from file or URL
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        results = extractor.extract_meta_information(html_content, args.url or "http://example.com")
    elif args.url:
        import requests
        response = requests.get(args.url)
        if response.status_code == 200:
            results = extractor.extract_meta_information(response.text, args.url)
        else:
            results = {"error": f"Failed to fetch URL: HTTP {response.status_code}"}
    else:
        results = {"error": "No input specified"}
        
    save_results(results, args.output)

def demo_serp_analyzer(args):
    """Run SERPAnalyzer demo."""
    analyzer = SERPAnalyzer(search_engine=args.search_engine)
    
    if args.query:
        results = analyzer.analyze_top_results(args.query)
    else:
        results = {"error": "No query specified"}
        
    save_results(results, args.output)

def demo_parsing_pipeline(args):
    """Run ParsingPipeline demo."""
    pipeline = ParsingPipeline(
        delay=args.delay,
        respect_robots=not args.ignore_robots,
        search_engine=args.search_engine
    )
    
    if args.mode == "url":
        if args.url:
            results = pipeline.analyze_url(args.url)
        else:
            results = {"error": "No URL specified"}
            
    elif args.mode == "site":
        if args.url:
            results = pipeline.crawl_and_analyze_site(
                args.url, 
                max_pages=args.max_pages
            )
        else:
            results = {"error": "No URL specified"}
            
    elif args.mode == "keyword":
        if args.query:
            results = pipeline.analyze_keyword(
                args.query,
                analyze_competitors=not args.skip_competitors
            )
        else:
            results = {"error": "No query specified"}
            
    else:
        results = {"error": f"Invalid mode: {args.mode}"}
        
    save_results(results, args.output)

def main():
    parser = argparse.ArgumentParser(description="SEO AI Models Parser Demonstrations")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Web Crawler demo
    crawler_parser = subparsers.add_parser("crawler", help="Run WebCrawler demo")
    crawler_parser.add_argument("url", help="URL to crawl")
    crawler_parser.add_argument("--max-pages", type=int, default=10, help="Maximum pages to crawl")
    crawler_parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests")
    crawler_parser.add_argument("--ignore-robots", action="store_true", help="Ignore robots.txt")
    crawler_parser.add_argument("--output", help="Output file for results (JSON)")
    
    # Content Extractor demo
    content_parser = subparsers.add_parser("content", help="Run ContentExtractor demo")
    content_parser.add_argument("--url", help="URL to extract content from")
    content_parser.add_argument("--file", help="HTML file to extract content from")
    content_parser.add_argument("--output", help="Output file for results (JSON)")
    
    # Meta Extractor demo
    meta_parser = subparsers.add_parser("meta", help="Run MetaExtractor demo")
    meta_parser.add_argument("--url", help="URL to extract meta information from")
    meta_parser.add_argument("--file", help="HTML file to extract meta information from")
    meta_parser.add_argument("--output", help="Output file for results (JSON)")
    
    # SERP Analyzer demo
    serp_parser = subparsers.add_parser("serp", help="Run SERPAnalyzer demo")
    serp_parser.add_argument("--query", help="Search query")
    serp_parser.add_argument("--search-engine", choices=["google", "bing"], default="google", help="Search engine to use")
    serp_parser.add_argument("--output", help="Output file for results (JSON)")
    
    # Parsing Pipeline demo
    pipeline_parser = subparsers.add_parser("pipeline", help="Run ParsingPipeline demo")
    pipeline_parser.add_argument("--mode", choices=["url", "site", "keyword"], required=True, help="Analysis mode")
    pipeline_parser.add_argument("--url", help="URL to analyze (for url and site modes)")
    pipeline_parser.add_argument("--query", help="Search query (for keyword mode)")
    pipeline_parser.add_argument("--max-pages", type=int, default=10, help="Maximum pages to crawl (for site mode)")
    pipeline_parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests")
    pipeline_parser.add_argument("--ignore-robots", action="store_true", help="Ignore robots.txt")
    pipeline_parser.add_argument("--search-engine", choices=["google", "bing"], default="google", help="Search engine to use")
    pipeline_parser.add_argument("--skip-competitors", action="store_true", help="Skip competitor analysis (for keyword mode)")
    pipeline_parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    if args.command == "crawler":
        demo_web_crawler(args)
    elif args.command == "content":
        demo_content_extractor(args)
    elif args.command == "meta":
        demo_meta_extractor(args)
    elif args.command == "serp":
        demo_serp_analyzer(args)
    elif args.command == "pipeline":
        demo_parsing_pipeline(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
