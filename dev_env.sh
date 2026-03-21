#!/bin/bash

set -e

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "Installing opencode..."
curl -fsSL https://opencode.ai/install | bash

echo "Installing huggingface-cli..."
uv pip install -U "huggingface_hub[cli]"

echo "Installing gh CLI..."
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update && sudo apt install -y gh

echo "Installation complete!"
echo ""
echo "REMINDERS:"
echo "- Update OpenRouter key in .env or config"
echo "- Run 'gh login' to authenticate with GitHub"
echo "- Set opencode API key: 'opencode config set api_key YOUR_KEY'"