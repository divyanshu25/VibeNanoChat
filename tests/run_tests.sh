#!/bin/bash
# Script to run VibeNanoChat tests with common options

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Determine the tests directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" == "tests" ]]; then
    # Script is in tests directory
    TESTS_DIR="$SCRIPT_DIR"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
else
    # Script might be called from elsewhere
    TESTS_DIR="$SCRIPT_DIR/tests"
    PROJECT_ROOT="$SCRIPT_DIR"
fi

echo -e "${BLUE}Running VibeNanoChat Test Suite${NC}\n"
echo -e "${BLUE}Project root: ${PROJECT_ROOT}${NC}"
echo -e "${BLUE}Tests directory: ${TESTS_DIR}${NC}\n"

# Change to project root for pytest to find src/
cd "$PROJECT_ROOT"

# Default: run all tests with verbose output
if [ $# -eq 0 ]; then
    echo -e "${GREEN}Running all tests...${NC}"
    uv run pytest "$TESTS_DIR" -v
else
    case "$1" in
        unit)
            echo -e "${GREEN}Running unit tests only...${NC}"
            uv run pytest "$TESTS_DIR/unit" -v
            ;;
        integration)
            echo -e "${GREEN}Running integration tests only...${NC}"
            uv run pytest "$TESTS_DIR/integration" -v
            ;;
        dataloaders)
            echo -e "${GREEN}Running dataloader tests...${NC}"
            uv run pytest "$TESTS_DIR/unit/dataloaders" -v
            ;;
        coverage)
            echo -e "${GREEN}Running tests with coverage report...${NC}"
            uv run pytest "$TESTS_DIR" --cov=src --cov-report=html --cov-report=term
            echo -e "\n${BLUE}Coverage report generated in htmlcov/index.html${NC}"
            ;;
        fast)
            echo -e "${GREEN}Running tests in parallel...${NC}"
            uv run pytest "$TESTS_DIR" -n auto
            ;;
        debug)
            echo -e "${GREEN}Running tests with debugger enabled...${NC}"
            uv run pytest "$TESTS_DIR" -v -s --pdb
            ;;
        failed)
            echo -e "${GREEN}Running previously failed tests...${NC}"
            uv run pytest "$TESTS_DIR" --lf -v
            ;;
        *)
            echo -e "${GREEN}Running specific test: $1${NC}"
            # Handle both absolute and relative paths
            if [[ "$1" = /* ]]; then
                # Absolute path
                uv run pytest "$1" -v
            else
                # Relative path - assume it's relative to tests directory
                uv run pytest "$TESTS_DIR/$1" -v
            fi
            ;;
    esac
fi

# Show summary
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ All tests passed!${NC}"
else
    echo -e "\n${RED}✗ Some tests failed${NC}"
    exit 1
fi
