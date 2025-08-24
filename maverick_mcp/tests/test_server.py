"""
Integration tests for the Maverick-MCP server.
"""

import json
import os
import subprocess
import time
import unittest
from typing import Any

import pytest
import requests

# Constants
SERVER_URL = "http://localhost:8000"
SERVER_START_TIMEOUT = 10  # seconds


@pytest.mark.integration
class TestMaverickMCPServer(unittest.TestCase):
    """Integration tests for the Maverick-MCP server."""

    process: subprocess.Popen[bytes] | None = None

    @classmethod
    def setUpClass(cls):
        """Start the server before running tests."""
        # Skip server startup if USE_RUNNING_SERVER environment variable is set
        cls.process = None
        if os.environ.get("USE_RUNNING_SERVER") != "1":
            print("Starting Maverick-MCP server...")
            # Start the server as a subprocess
            cls.process = subprocess.Popen(
                ["python", "-m", "maverick_mcp.api.server"],
                # Redirect stdout and stderr to prevent output in test results
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for the server to start
            start_time = time.time()
            while time.time() - start_time < SERVER_START_TIMEOUT:
                try:
                    response = requests.get(f"{SERVER_URL}/health")
                    if response.status_code == 200:
                        print("Server started successfully")
                        break
                except requests.exceptions.ConnectionError:
                    pass
                time.sleep(0.5)
            else:
                # If the server didn't start within the timeout, kill it and fail
                cls.tearDownClass()
                raise TimeoutError("Server did not start within the timeout period")

    @classmethod
    def tearDownClass(cls):
        """Stop the server after tests are done."""
        if cls.process:
            print("Stopping Maverick-MCP server...")
            # Send SIGTERM signal to the process
            cls.process.terminate()
            try:
                # Wait for the process to terminate
                cls.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # If the process doesn't terminate within 5 seconds, kill it
                cls.process.kill()
                cls.process.wait()

    def test_health_endpoint(self):
        """Test the health endpoint."""
        response = requests.get(f"{SERVER_URL}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        # Version should be present
        self.assertIn("version", data)

    def test_mcp_endpoint(self):
        """Test the MCP endpoint."""
        # This is a simple request to test if the MCP endpoint is responding
        sse_url = f"{SERVER_URL}/sse"
        response = requests.get(sse_url)
        # Just check that the endpoint exists and responds with success
        self.assertIn(
            response.status_code, [200, 405]
        )  # 200 OK or 405 Method Not Allowed

    def send_mcp_request(self, method: str, params: list[Any]) -> dict[str, Any]:
        """
        Send a request to the MCP server.

        Args:
            method: The method name
            params: The parameters for the method

        Returns:
            The response from the server
        """
        request_body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}

        response = requests.post(
            f"{SERVER_URL}/messages/",
            json=request_body,
            headers={"Content-Type": "application/json"},
        )

        # Check that the request was successful
        self.assertEqual(response.status_code, 200)

        # Parse the response
        data = response.json()

        # Check that the response is valid JSON-RPC
        self.assertEqual(data["jsonrpc"], "2.0")
        self.assertEqual(data["id"], 1)

        return data  # type: ignore[no-any-return]

    def test_fetch_stock_data(self):
        """Test the fetch_stock_data tool."""
        # Send a request to fetch stock data for a known symbol (AAPL)
        response_data = self.send_mcp_request("fetch_stock_data", ["AAPL"])

        # Check that the result is present and contains stock data
        self.assertIn("result", response_data)
        result = response_data["result"]

        # Parse the result as JSON
        stock_data = json.loads(result)

        # Check that the stock data contains the expected fields
        self.assertIn("index", stock_data)
        self.assertIn("columns", stock_data)
        self.assertIn("data", stock_data)

        # Check that the columns include OHLCV
        for column in ["open", "high", "low", "close", "volume"]:
            self.assertIn(
                column.lower(), [col.lower() for col in stock_data["columns"]]
            )


# Run the tests if this script is executed directly
if __name__ == "__main__":
    unittest.main()
