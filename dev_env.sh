#!/bin/bash

set -e

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "Installing opencode..."
curl -fsSL https://opencode.ai/install | bash

echo "Installing huggingface-cli..."
curl -LsSf https://hf.co/cli/install.sh | bash


echo "Installation complete!"
echo ""
echo "REMINDERS:"
echo "- Update OpenRouter key in .env or config"
echo "- Run 'gh login' to authenticate with GitHub"
echo "- Set opencode API key: 'opencode config set api_key YOUR_KEY'"
