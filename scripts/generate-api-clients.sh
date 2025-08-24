#!/bin/bash
# Generate API clients from OpenAPI specification

set -e

echo "🚀 Generating API clients from OpenAPI specification..."

# Check if openapi-generator-cli is installed
if ! command -v openapi-generator-cli &> /dev/null; then
    echo "❌ openapi-generator-cli is not installed."
    echo "📦 Please install openapi-generator-cli:"
    echo "   Using pip: pip install openapi-generator-cli"
    echo "   Using brew: brew install openapi-generator"
    exit 1
fi

# Check if API server is running
if ! curl -s http://localhost:8000/api/openapi.json > /dev/null; then
    echo "❌ API server is not running on http://localhost:8000"
    echo "Please start the API server first with: ./scripts/start-backend-api.sh"
    exit 1
fi

# Download latest OpenAPI spec
echo "📥 Downloading OpenAPI specification..."
curl -s http://localhost:8000/api/openapi.json -o openapi-spec.json

# Create output directory
mkdir -p generated/python-client

# Generate Python client
echo "🐍 Generating Python client..."
openapi-generator-cli generate \
    -c openapi-generator-config-python.yaml \
    -i openapi-spec.json \
    --skip-validate-spec

# Clean up
rm -f openapi-spec.json

echo "✅ Python API client generated successfully!"
echo ""
echo "📂 Python client: ./generated/python-client"
echo ""
echo "To use the Python client:"
echo "  cd generated/python-client"
echo "  pip install -e ."