# Installation Guide for miniManus

This guide provides detailed instructions for installing and setting up miniManus on your Android device using Termux.

## Prerequisites

- Android device (Android 7.0 or higher recommended)
- [Termux](https://f-droid.org/en/packages/com.termux/) installed from F-Droid
- At least 500MB of free storage space
- Internet connection for initial setup and API access

## Basic Installation

### Step 1: Install Termux

1. Download and install Termux from F-Droid: [https://f-droid.org/en/packages/com.termux/](https://f-droid.org/en/packages/com.termux/)
2. Open Termux and wait for the initial setup to complete

### Step 2: Update Termux Packages

```bash
pkg update && pkg upgrade -y
```

### Step 3: Install Required Packages

```bash
pkg install -y python git wget curl openssh
```

### Step 4: Clone the miniManus Repository

```bash
cd ~
git clone https://github.com/yourusername/miniManus.git
cd miniManus
```

### Step 5: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run miniManus

```bash
python -m minimanus
```

## Advanced Installation Options

### Installing from ZIP Archive

If you prefer not to use Git, you can download and extract a ZIP archive:

```bash
cd ~
wget https://github.com/yourusername/miniManus/archive/main.zip
unzip main.zip
mv miniManus-main miniManus
cd miniManus
pip install -r requirements.txt
```

### Creating a Launcher Script

For easier access, create a launcher script:

```bash
echo '#!/data/data/com.termux/files/usr/bin/bash
cd ~/miniManus
python -m minimanus "$@"' > ~/.termux/bin/minimanus
chmod +x ~/.termux/bin/minimanus
```

Now you can start miniManus by simply typing `minimanus` in Termux.

## Configuration

### Setting Up API Keys

To use external API providers, you'll need to configure API keys:

#### OpenRouter

1. Create an account at [OpenRouter](https://openrouter.ai/)
2. Generate an API key
3. In miniManus, go to Settings > API > OpenRouter and enter your key

#### DeepSeek

1. Create an account at [DeepSeek](https://deepseek.ai/)
2. Generate an API key
3. In miniManus, go to Settings > API > DeepSeek and enter your key

#### Anthropic

1. Create an account at [Anthropic](https://www.anthropic.com/)
2. Generate an API key
3. In miniManus, go to Settings > API > Anthropic and enter your key

### Configuring Local Models with Ollama

To use Ollama with miniManus:

1. Install Ollama on a device in your local network:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. Start Ollama:
   ```bash
   ollama serve
   ```

3. In miniManus, go to Settings > API > Ollama
4. The automatic discovery should find your Ollama server
5. If not, manually enter the IP address and port

### Configuring LiteLLM

To use LiteLLM with miniManus:

1. Install LiteLLM on a device in your local network:
   ```bash
   pip install litellm
   ```

2. Start LiteLLM proxy:
   ```bash
   litellm --model ollama/llama2 --port 8000
   ```

3. In miniManus, go to Settings > API > LiteLLM
4. The automatic discovery should find your LiteLLM server
5. If not, manually enter the IP address and port

## Troubleshooting

### Common Issues

#### Package Installation Failures

If you encounter package installation failures, try:

```bash
pkg clean
pkg update -y
pkg upgrade -y
```

Then retry the installation.

#### Permission Issues

If you encounter permission issues:

```bash
chmod -R 755 ~/miniManus
```

#### Network Discovery Not Working

If automatic network discovery for Ollama or LiteLLM isn't working:

1. Ensure both devices are on the same network
2. Check if any firewall is blocking the ports (11434 for Ollama, 8000 for LiteLLM)
3. Try manually entering the server IP and port

#### High Memory Usage

If miniManus is using too much memory:

1. Go to Settings > Advanced
2. Reduce the "Max Memory (MB)" setting
3. Enable "Compact Mode" in Settings > Appearance

### Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/yourusername/miniManus/issues) page
2. Join our [Discord community](https://discord.gg/yourdiscord)
3. Submit a new issue with details about your problem

## Updating miniManus

To update miniManus to the latest version:

```bash
cd ~/miniManus
git pull
pip install -r requirements.txt
```

## Uninstalling miniManus

To completely remove miniManus:

```bash
rm -rf ~/miniManus
rm -rf ~/.local/share/minimanus
```
