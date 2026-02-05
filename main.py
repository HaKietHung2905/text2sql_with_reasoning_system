#!/usr/bin/env python3
"""
Main script to download and setup Spider dataset for Text2SQL tasks.
Run this from the project root directory.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import from utils
from utils.others.spider_downloader import SpiderDatasetDownloader

def main():
    """Main function - all logic is now in spider_downloader.py"""
    # Import and run the main function from spider_downloader
    from utils.others.spider_downloader import main as spider_main
    return spider_main()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)