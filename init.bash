#!/bin/bash

# Check if virtual environment is already activated
if [ -z "$VIRTUAL_ENV" ]; then
    python3 -m venv venv
    source venv/bin/activate
fi

pip install -r requirements.txt