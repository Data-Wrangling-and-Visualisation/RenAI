#!/bin/bash

set -e

PYTHON_MAJOR_VERSION="3"
PYTHON_MINOR_REQUIRED_MIN="10"
PYTHON_MINOR_REQUIRED_MAX="11"
SERVER_DIR="server"
FRONTEND_DIR="react-renai"
VENV_NAME="venv"

check_command() {
  if ! command -v "$1" &> /dev/null; then
    echo "Error: Required command '$1' not found. Please install it."
    exit 1
  fi
}

check_python_version() {
  echo "Checking Python version..."
  check_command python3
  PYTHON_VERSION_STRING=$(python3 --version 2>&1)
  if [[ $PYTHON_VERSION_STRING =~ Python\ ([0-9]+)\.([0-9]+)\.?([0-9]*) ]]; then
      PY_MAJOR=${BASH_REMATCH[1]}
      PY_MINOR=${BASH_REMATCH[2]}
      echo "Found Python version: $PY_MAJOR.$PY_MINOR"
      if [[ "$PY_MAJOR" -ne "$PYTHON_MAJOR_VERSION" ]] || \
         [[ "$PY_MINOR" -lt "$PYTHON_MINOR_REQUIRED_MIN" ]] || \
         [[ "$PY_MINOR" -gt "$PYTHON_MINOR_REQUIRED_MAX" ]]; then
          echo "Error: Incorrect Python version. Found $PY_MAJOR.$PY_MINOR, but require ${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_REQUIRED_MIN} or ${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_REQUIRED_MAX}."
          echo "Please install the correct Python version and ensure 'python3' points to it."
          exit 1
      fi
  else
      echo "Error: Could not parse Python version string: '$PYTHON_VERSION_STRING'"
      exit 1
  fi
}

echo "--- Checking Prerequisites ---"
check_python_version
check_command node
check_command npm
echo "Prerequisites met."
echo

echo "--- Setting up Backend ($SERVER_DIR) ---"
if [ ! -d "$SERVER_DIR" ]; then
    echo "Error: Server directory '$SERVER_DIR' not found in current location ($(pwd))."
    exit 1
fi
cd "$SERVER_DIR"

echo "Creating/Activating Python virtual environment ($VENV_NAME)..."
if [ ! -d "$VENV_NAME" ]; then
    python3 -m venv "$VENV_NAME"
    echo "Virtual environment created."
fi
# shellcheck source=/dev/null
source "$VENV_NAME/bin/activate"
echo "Virtual environment activated."

echo "Installing Python requirements from requirements.txt..."
pip install -r requirements.txt
echo "Backend dependencies installed."
cd ..
echo

echo "--- Setting up Frontend ($FRONTEND_DIR) ---"
 if [ ! -d "$FRONTEND_DIR" ]; then
    echo "Error: Frontend directory '$FRONTEND_DIR' not found in current location ($(pwd))."
    exit 1
fi
cd "$FRONTEND_DIR"

echo "Installing Node dependencies (npm install)..."
npm install
echo "Frontend dependencies installed."
cd ..
echo

echo "--- Starting Application ---"

# shellcheck source=/dev/null
source "$SERVER_DIR/$VENV_NAME/bin/activate"

echo "Starting Backend Server (server/api.py) in the background..."
export FLASK_ENV=development
python3 "$SERVER_DIR/api.py" &
BACKEND_PID=$!
echo "Backend server started with PID: $BACKEND_PID"
sleep 5

cleanup() {
    echo "Shutting down script..."
    echo "Killing backend server (PID: $BACKEND_PID)..."
    kill -TERM $BACKEND_PID 2>/dev/null
    echo "Backend server stopped."
    if type deactivate &> /dev/null; then
        deactivate
    fi
}

trap cleanup EXIT INT TERM

echo "Starting Frontend Development Server (npm run dev)..."
cd "$FRONTEND_DIR"
npm run dev

echo "Frontend server stopped."
