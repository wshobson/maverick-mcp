#!/usr/bin/env python3
"""
Example script demonstrating the Maverick-MCP message queue system.

This script shows how to:
1. Submit async jobs
2. Monitor job progress
3. Retrieve results
4. Handle errors

Run this script with: python examples/queue_system_example.py
"""

import time
from typing import Any

import requests
from requests import RequestException

# Configuration
API_BASE_URL = "http://localhost:8000/api"
DEMO_USER_TOKEN = None  # Set if authentication is enabled


class QueueSystemDemo:
    """Demonstration of the queue system capabilities."""

    def __init__(self, base_url: str = API_BASE_URL, token: str | None = None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def submit_job(
        self, job_type: str, job_name: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Submit an async job."""
        url = f"{self.base_url}/jobs/submit"
        payload = {
            "job_type": job_type,
            "job_name": job_name,
            "parameters": parameters,
            "priority": "normal",
        }

        print(f"🚀 Submitting {job_type} job: {job_name}")
        print(f"   Parameters: {parameters}")

        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            job_data = response.json()

            print("✅ Job submitted successfully!")
            print(f"   Job ID: {job_data['id']}")
            print(f"   Credits reserved: {job_data['credits_reserved']}")
            print(
                f"   Estimated duration: {job_data.get('estimated_duration', 'Unknown')} seconds"
            )

            return job_data
        except RequestException as e:
            print(f"❌ Failed to submit job: {e}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"   Error details: {error_detail}")
                except Exception:
                    print(f"   Response text: {e.response.text}")
            return {}

    def check_job_status(self, job_id: str) -> dict[str, Any]:
        """Check the status of a job."""
        url = f"{self.base_url}/jobs/{job_id}"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            print(f"❌ Failed to check job status: {e}")
            return {}

    def wait_for_completion(self, job_id: str, timeout: int = 300) -> dict[str, Any]:
        """Wait for job completion with progress updates."""
        print(f"⏳ Waiting for job {job_id} to complete...")

        start_time = time.time()
        last_progress = -1

        while time.time() - start_time < timeout:
            status_data = self.check_job_status(job_id)

            if not status_data:
                time.sleep(5)
                continue

            current_status = status_data.get("status", "unknown")
            current_progress = status_data.get("progress_percent", 0)
            status_message = status_data.get("status_message", "")

            # Show progress updates
            if current_progress != last_progress:
                print(f"   📊 Progress: {current_progress:.1f}% - {status_message}")
                last_progress = current_progress

            # Check if job is completed
            if current_status in ["success", "failure", "cancelled"]:
                if current_status == "success":
                    print("✅ Job completed successfully!")
                    credits_consumed = status_data.get("credits_consumed", 0)
                    if credits_consumed:
                        print(f"   Credits consumed: {credits_consumed}")
                elif current_status == "failure":
                    print("❌ Job failed!")
                    error_msg = status_data.get("error_message", "Unknown error")
                    print(f"   Error: {error_msg}")
                else:
                    print("⚠️ Job was cancelled")

                return status_data

            time.sleep(2)  # Poll every 2 seconds

        print(f"⏰ Job timed out after {timeout} seconds")
        return self.check_job_status(job_id)

    def get_job_result(self, job_id: str) -> dict[str, Any]:
        """Get the result of a completed job."""
        url = f"{self.base_url}/jobs/{job_id}/result"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            print(f"❌ Failed to get job result: {e}")
            return {}

    def demo_screening_job(self):
        """Demonstrate a stock screening job."""
        print("\n" + "=" * 60)
        print("📈 DEMO: Maverick Stock Screening")
        print("=" * 60)

        # Submit screening job
        job_data = self.submit_job(
            job_type="maverick_screening",
            job_name="Demo Maverick Screening",
            parameters={"limit": 10, "min_score": 80},
        )

        if not job_data:
            return

        job_id = job_data["id"]

        # Wait for completion
        final_status = self.wait_for_completion(job_id)

        if final_status.get("status") == "success":
            # Get results
            result_data = self.get_job_result(job_id)
            if result_data:
                stocks = result_data.get("result_data", {}).get("stocks", [])
                print(f"\n📊 Found {len(stocks)} high-momentum stocks:")
                for i, stock in enumerate(stocks[:5], 1):  # Show top 5
                    symbol = stock.get("symbol", "N/A")
                    score = stock.get("combined_score", 0)
                    price = stock.get("close", 0)
                    print(f"   {i}. {symbol}: Score {score:.1f}, Price ${price:.2f}")

    def demo_portfolio_analysis(self):
        """Demonstrate portfolio correlation analysis."""
        print("\n" + "=" * 60)
        print("📊 DEMO: Portfolio Correlation Analysis")
        print("=" * 60)

        # Submit portfolio job
        job_data = self.submit_job(
            job_type="portfolio_correlation",
            job_name="Demo Portfolio Analysis",
            parameters={"tickers": ["AAPL", "GOOGL", "MSFT", "TSLA"], "days": 90},
        )

        if not job_data:
            return

        job_id = job_data["id"]

        # Wait for completion
        final_status = self.wait_for_completion(job_id)

        if final_status.get("status") == "success":
            # Get results
            result_data = self.get_job_result(job_id)
            if result_data:
                analysis = result_data.get("result_data", {})
                avg_correlation = analysis.get("average_portfolio_correlation", 0)
                diversification_score = analysis.get("diversification_score", 0)

                print("\n📊 Portfolio Analysis Results:")
                print(f"   Average correlation: {avg_correlation:.3f}")
                print(f"   Diversification score: {diversification_score:.1f}")
                print(f"   Recommendation: {analysis.get('recommendation', 'N/A')}")

                # Show high correlation pairs
                high_corr_pairs = analysis.get("high_correlation_pairs", [])
                if high_corr_pairs:
                    print("\n⚠️ High correlation pairs:")
                    for pair in high_corr_pairs[:3]:  # Show top 3
                        stocks = pair["pair"]
                        correlation = pair["correlation"]
                        print(f"   {stocks[0]} - {stocks[1]}: {correlation:.3f}")

    def demo_comprehensive_screening(self):
        """Demonstrate comprehensive screening across all strategies."""
        print("\n" + "=" * 60)
        print("🔍 DEMO: Comprehensive Multi-Strategy Screening")
        print("=" * 60)

        # Submit comprehensive screening job
        job_data = self.submit_job(
            job_type="comprehensive_screening",
            job_name="Demo Comprehensive Screening",
            parameters={"include_bear": True},
        )

        if not job_data:
            return

        job_id = job_data["id"]

        # Wait for completion
        final_status = self.wait_for_completion(job_id, timeout=180)  # Longer timeout

        if final_status.get("status") == "success":
            # Get results
            result_data = self.get_job_result(job_id)
            if result_data:
                analysis = result_data.get("result_data", {})
                summary = analysis.get("summary", {})

                print("\n📊 Comprehensive Screening Results:")
                print(f"   Total stocks found: {summary.get('total_stocks_found', 0)}")
                print(f"   Strategies analyzed: {summary.get('strategies_run', 0)}")
                print(
                    f"   Maverick bullish: {summary.get('maverick_bullish_count', 0)}"
                )
                print(f"   Trending stocks: {summary.get('trending_count', 0)}")
                print(
                    f"   Maverick bearish: {summary.get('maverick_bearish_count', 0)}"
                )

    def demo_error_handling(self):
        """Demonstrate error handling."""
        print("\n" + "=" * 60)
        print("❌ DEMO: Error Handling")
        print("=" * 60)

        # Submit job with invalid parameters
        print("Testing invalid job type...")
        invalid_job = self.submit_job(
            job_type="invalid_job_type", job_name="Invalid Job Demo", parameters={}
        )

        if invalid_job:
            print("❌ Unexpected: Invalid job was accepted")
        else:
            print("✅ Invalid job correctly rejected")

        # Test job with missing required parameters
        print("\nTesting job with invalid parameters...")
        invalid_params_job = self.submit_job(
            job_type="portfolio_correlation",
            job_name="Invalid Params Demo",
            parameters={
                "tickers": ["AAPL"]  # Need at least 2 tickers
            },
        )

        if invalid_params_job:
            job_id = invalid_params_job["id"]
            final_status = self.wait_for_completion(job_id, timeout=60)

            if final_status.get("status") == "failure":
                print("✅ Job correctly failed with invalid parameters")
            else:
                print("❌ Job should have failed but didn't")

    def list_recent_jobs(self):
        """List recent jobs for demonstration."""
        print("\n" + "=" * 60)
        print("📋 Recent Jobs")
        print("=" * 60)

        url = f"{self.base_url}/jobs?page=1&page_size=10"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            jobs_data = response.json()

            jobs = jobs_data.get("jobs", [])
            if jobs:
                print(f"Found {len(jobs)} recent jobs:")
                for i, job in enumerate(jobs, 1):
                    job_type = job.get("job_type", "unknown")
                    status = job.get("status", "unknown")
                    created_at = job.get("created_at", "unknown")
                    progress = job.get("progress_percent", 0)

                    print(
                        f"   {i}. {job_type} - {status} ({progress:.1f}%) - {created_at}"
                    )
            else:
                print("No recent jobs found.")
        except RequestException as e:
            print(f"❌ Failed to list jobs: {e}")

    def run_all_demos(self):
        """Run all demonstration scenarios."""
        print("🎯 Starting Maverick-MCP Queue System Demonstration")
        print("This demo will showcase async job processing capabilities.")

        # Check if API is available
        try:
            response = requests.get(
                f"{self.base_url.replace('/api', '')}/health-simple"
            )
            if response.status_code != 200:
                print("❌ API server not available. Make sure the server is running.")
                return
        except Exception:
            print(
                "❌ Cannot connect to API server. Make sure it's running on the correct port."
            )
            return

        # Run demos
        self.demo_screening_job()
        self.demo_portfolio_analysis()
        self.demo_comprehensive_screening()
        self.demo_error_handling()
        self.list_recent_jobs()

        print("\n" + "=" * 60)
        print("✨ Demo completed!")
        print("=" * 60)
        print("Key takeaways:")
        print("• Jobs run asynchronously without blocking the API")
        print("• Real-time progress updates keep users informed")
        print("• Credit system tracks resource consumption")
        print("• Comprehensive error handling ensures reliability")
        print("• Multiple job types support different use cases")
        print("\nFor production use:")
        print("• Start workers: make worker")
        print("• Monitor with Flower: make flower")
        print("• Check queue stats: make queue-stats")


def main():
    """Main demonstration function."""
    demo = QueueSystemDemo(token=DEMO_USER_TOKEN)

    print("Maverick-MCP Queue System Demo")
    print("==============================")
    print()
    print("This script demonstrates the async job processing system.")
    print("Make sure the following are running:")
    print("1. Redis server (make redis-start)")
    print("2. MCP API server (make backend)")
    print("3. Celery worker (make worker)")
    print()

    response = input("Ready to start the demo? (y/n): ").strip().lower()
    if response == "y":
        demo.run_all_demos()
    else:
        print("Demo cancelled.")


if __name__ == "__main__":
    main()
