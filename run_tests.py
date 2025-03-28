#!/usr/bin/env python3
import os
import sys
import time
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results/test_run.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_tests():
    """Run the test suite and capture results."""
    # Create test results directory if it doesn't exist
    Path('test_results').mkdir(exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    try:
        # Run pytest with coverage
        logging.info("Starting test suite...")
        result = subprocess.run(
            [
                'pytest',
                'tests/',
                '--cov=src',
                '--cov-report=html',
                '--cov-report=term-missing',
                '-v'
            ],
            capture_output=True,
            text=True
        )
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log results
        logging.info(f"Test suite completed in {duration:.2f} seconds")
        
        # Write detailed output to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'test_results/test_output_{timestamp}.txt'
        
        with open(output_file, 'w') as f:
            f.write(f"Test Run: {datetime.now()}\n")
            f.write(f"Duration: {duration:.2f} seconds\n")
            f.write("\n=== Test Output ===\n")
            f.write(result.stdout)
            f.write("\n=== Error Output ===\n")
            f.write(result.stderr)
        
        # Check if tests passed
        if result.returncode == 0:
            logging.info("All tests passed successfully!")
        else:
            logging.error("Some tests failed. Check the detailed output file.")
            
        # Print summary
        print("\nTest Summary:")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Output file: {output_file}")
        print(f"Coverage report: test_results/htmlcov/index.html")
        
    except Exception as e:
        logging.error(f"Error running tests: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    run_tests() 