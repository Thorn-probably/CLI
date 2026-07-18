# CLI: Dynamic Provider & Explicit ID Indexing Chat

A highly flexible, provider-agnostic terminal chat application written in Python.

**CLI** features an interactive setup dialog for various AI providers, persistent configuration, dynamic server-side model fetching, local context indexing via an Ollama Librarian, a Deep Search Matrix, deduplicated Long-Term Memory (RAG) injection, and explicit state tags.

## 🌟 Features

- **Provider-Agnostic**: Connect to Groq, Google, Z.ai, OpenRouter, DeepSeek, or any OpenAI-compatible API.
- **Long-Term Memory (RAG)**: Deduplicated and context-aware memory using a local Ollama server.
- **Deep Search Matrix**: Instantly scan through saved chat logs for specific keywords and context.
- **Profile Management**: Save and swap between different provider and model configurations on the fly.
- **Persistent Sessions**: Autosaves your conversation and allows manual save/load capabilities.
- **Rich Terminal UI**: ANSI colored output and formatted history.

---

## ⚙️ Prerequisites

To run **CLI**, you'll need the following installed on your system:

1. **Python 3.7+**
2. **Ollama** (Required for the local `/remember` memory deduplication feature)
   - Ensure you have the `qwen2.5:3b` model pulled in Ollama for the memory engine to work:
     ```bash
     ollama run qwen2.5:3b
     ```

---

## 🚀 Installation & Setup

### 🪟 Windows Setup

1. **Clone the repository or download the source code**:
   ```cmd
   git clone <your-repo-url>
   cd CLI
   ```

2. **Install required dependencies**:
   ```cmd
   pip install requests
   ```

3. **Set your API keys as Environment Variables**:
   By default, the app looks for specific environment variables. For example, to use Groq:
   ```cmd
   setx groq_api1 "YOUR_GROQ_API_KEY"
   setx googleai_api1 "YOUR_GOOGLE_API_KEY"
   ```
   *(You may need to restart your terminal after using `setx` for the variables to take effect.)*

4. **Run the CLI**:
   ```cmd
   python cli.py
   ```

### 🍎 macOS Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd CLI
   ```

2. **Install required dependencies**:
   ```bash
   pip3 install requests
   ```

3. **Set your API keys as Environment Variables**:
   Add the following to your `~/.zshrc` or `~/.bash_profile`:
   ```bash
   export groq_api1="YOUR_GROQ_API_KEY"
   export googleai_api1="YOUR_GOOGLE_API_KEY"
   ```
   Then reload your profile: `source ~/.zshrc`

4. **Run the CLI**:
   ```bash
   python3 cli.py
   ```

### 🐧 Linux Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd CLI
   ```

2. **Install required dependencies**:
   ```bash
   pip3 install requests
   ```

3. **Set your API keys as Environment Variables**:
   Add the following to your `~/.bashrc` or `~/.zshrc`:
   ```bash
   export groq_api1="YOUR_GROQ_API_KEY"
   export googleai_api1="YOUR_GOOGLE_API_KEY"
   ```
   Then reload your profile: `source ~/.bashrc`

4. **Run the CLI**:
   ```bash
   python3 cli.py
   ```
   *(Optionally, you can make the script executable by running `chmod +x cli.py` and then executing it with `./cli.py`)*

---

## 🎮 Usage & Commands

Once you launch the application, you can chat directly with the active model. Use the following slash commands to interact with the system's features:

### General Commands
- `/help` - View all available commands.
- `/clear` - Wipe the active conversation memory from the screen.
- `/exit` - Close the application safely.

### Provider & Model Management
- `/provider` - List active and available API providers.
- `/provider add` - Start an interactive setup dialog to add a new provider (e.g., OpenRouter).
- `/models` - Fetch and list all available models from your active provider.
- `/switch <provider> <model>` - Switch provider and model simultaneously.

### Profiles
- `/profile save <name>` - Save your current provider & model config.
- `/profile switch <name>` - Switch to a previously saved profile.

### Memory & Search (RAG)
- `/remember <fact>` - Summarize, deduplicate, and store a fact in the Long-Term Memory Matrix.
- `/memories` - View all saved long-term memories.
- `/forget <M#>` - Delete a specific memory (e.g., `/forget M1`).
- `/search <word>` - Scan all saved chats for a keyword.

### Session Management
- `/save <filename>` - Save the current conversation state.
- `/load <filename>` - Load a previously saved conversation.
- `/list` - List all saved conversation files.

### Customization
- `/name user <name>` - Change your display name.
- `/name bot <name>` - Change the assistant's name (Default is Assistant).
- `/detail <low|medium|high>` - Adjust the prompt verbosity of the assistant.

---

## 📁 Data Storage

All configurations, saved sessions, and memory indices are securely stored locally in your home directory:
`~/.cli-memory-matrix/`

Enjoy exploring the matrix!
