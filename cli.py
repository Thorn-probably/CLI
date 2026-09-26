#!/usr/bin/env python3
"""
AI Terminal Chat - Dynamic Provider & Explicit ID Indexing Edition
A highly flexible, provider-agnostic terminal chat application.
Features an interactive CLI setup dialog for providers, persistent configuration,
dynamic server-side model fetching, local context indexing via an Ollama Librarian,
a Deep Search Matrix, deduplicated Long-Term Memory (RAG) injection, and explicit state tags.
"""

import os
import sys
import json
import time
import datetime
import re
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: Missing required dependency 'requests'.")
    print("Please install it by running: pip install requests")
    sys.exit(1)

try:
    import readline
except ImportError:
    pass

# --- DYNAMIC CONFIGURATION CONSTANTS ---
HOME_DIR = str(Path.home())
BASE_DIR = os.path.join(HOME_DIR, ".cli-memory-matrix")
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
MEMORY_DIR = os.path.join(BASE_DIR, "memory")

CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
MEMORIES_PATH = os.path.join(MEMORY_DIR, "memories.json")
DEFAULT_AUTOSAVE_FILE = "_autosave.txt"

# ANSI Terminal Colors
COLORS = {
    "red": "\033[91m",
    "blue": "\033[94m",
    "yellow": "\033[93m",
    "green": "\033[92m",
    "cyan": "\033[96m",
    "magenta": "\033[95m",
    "white": "\033[97m",
    "reset": "\033[0m"
}

class ConfigManager:
    DEFAULT_CONFIG = {
        "provider": "groq",
        "model": "qwen/qwen3.6-27b",
        "assistant_name": "Assistant",
        "user_name": "User",
        "user_color": "cyan",
        "assistant_color": "red",
        "system_color": "yellow",
        "timestamps": True,
        "autosave": True,
        "detail_mode": "medium",
        "profiles": {},
        "providers": {
            "groq": {
                "url": "https://api.groq.com/openai/v1/chat/completions",
                "env_var": "groq_api1"
            },
            "google": {
                "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                "env_var": "googleai_api1"
            },
            "zai": {
                "url": "https://api.z.ai/api/paas/v4/chat/completions",
                "env_var": "zai_api1"
            }
        }
    }

    def __init__(self):
        self.config = self.DEFAULT_CONFIG.copy()
        self.ensure_directory()
        self.load_config()

    def ensure_directory(self):
        for directory in [BASE_DIR, SESSIONS_DIR, MEMORY_DIR]:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                except Exception as e:
                    print(f"{COLORS['red']}Failed to create directory {directory}: {e}{COLORS['reset']}")
                    sys.exit(1)

    def load_config(self):
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if "providers" in loaded and isinstance(loaded["providers"], dict):
                        self.config["providers"].update(loaded["providers"])
                        del loaded["providers"]
                    if "profiles" in loaded and isinstance(loaded["profiles"], dict):
                        self.config["profiles"].update(loaded["profiles"])
                        del loaded["profiles"]
                    for k, v in loaded.items():
                        if k in self.config:
                            self.config[k] = v
            except Exception:
                pass
        self.save_config()

    def save_config(self):
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
        except Exception:
            pass

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        if key in self.config:
            # stop cli over write of nested dicts so app does not crash
            if isinstance(self.config[key], dict):
                return False, f"Cant change complex structure '{key}' with cli. Please edit config.json."
                
            if isinstance(self.config[key], bool):
                if str(value).lower() in ['true', '1', 'yes']: value = True
                elif str(value).lower() in ['false', '0', 'no']: value = False
                else: return False, "Value must be a boolean (true/false)"
            if key.endswith('_color') and str(value).lower() not in COLORS:
                return False, f"Invalid color. Choose from: {', '.join(COLORS.keys())}"

            self.config[key] = value
            self.save_config()
            return True, f"{key} successfully updated."
        return False, f"Configuration key '{key}' does not exist."


class UniversalAIClient:
    def __init__(self, config_manager):
        self.config = config_manager
        self.api_key = None
        self.chat_endpoint = None
        self.update_provider(self.config.get("provider"))

    def update_provider(self, provider_name):
        providers = self.config.get("providers")
        if provider_name not in providers:
            raise ValueError(f"Unknown provider: {provider_name}")
            
        provider_data = providers[provider_name]
        self.chat_endpoint = provider_data["url"]
        env_var_name = provider_data["env_var"]
        self.api_key = os.environ.get(env_var_name)
        
        if not self.api_key:
            print(f"\033[91mWarning: Environment variable '{env_var_name}' is not set!\033[0m")

    def get_models(self):
        if not self.chat_endpoint or not self.api_key:
            return ["Error: Provider endpoint or API key is missing."]
        
        models_url = self.chat_endpoint.replace("/chat/completions", "/models") if self.chat_endpoint.endswith("/chat/completions") else self.chat_endpoint.rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.get(models_url, headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()
            if "data" in data and isinstance(data["data"], list):
                models = [model.get("id", "Unknown") for model in data["data"] if isinstance(model, dict) and "id" in model]
                return sorted(list(set(models)))
            return ["Error: Provider returned non-standard models JSON format."]
        except Exception as e:
            return [f"Network/API Error fetching models: {e}"]

    def stream_completion(self, model, history):
        if not self.api_key:
            yield f"\n[Error: Missing API key for this provider.]"
            return

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": history, "stream": True}

        try:
            response = requests.post(self.chat_endpoint, headers=headers, json=payload, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8').strip()
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str == "[DONE]": break
                        try:
                            chunk = json.loads(data_str)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta and delta["content"] is not None:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue
        except requests.exceptions.HTTPError as e:
            yield f"\n[API Error: {e.response.status_code} - {e.response.text}]"
        except requests.exceptions.RequestException as e:
            yield f"\n[Network Error: {e}]"


class ChatApplication:
    def __init__(self):
        self.config = ConfigManager()
        self.client = UniversalAIClient(self.config)
        self.system_prompt = self.build_system_prompt()
        self.o_counter = 0
        self.a_counter = 0
        self.m_counter = 0
        self.history = [self.system_prompt]
        self.memories = self.load_memories()
        self.current_file = DEFAULT_AUTOSAVE_FILE

    def _upgrade_history(self, history_data):
        o_count, a_count = 0, 0
        for msg in history_data:
            if msg.get("role") == "system": continue
            if "id" not in msg:
                if msg["role"] == "user":
                    o_count += 1
                    msg["id"] = f"O{o_count}"
                elif msg["role"] == "assistant":
                    a_count += 1
                    msg["id"] = f"A{a_count}"
            else:
                match = re.search(r'([OA])(\d+)', msg["id"])
                if match:
                    prefix, num = match.groups()
                    if prefix == 'O': o_count = max(o_count, int(num))
                    elif prefix == 'A': a_count = max(a_count, int(num))
        
        self.o_counter = o_count
        self.a_counter = a_count
        return history_data

    def load_memories(self):
        if os.path.exists(MEMORIES_PATH):
            try:
                with open(MEMORIES_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    upgraded_memories = []
                    max_id = 0
                    
                    for i, item in enumerate(data):
                        if isinstance(item, str):
                            m_id = f"M{i+1}"
                            upgraded_memories.append({
                                "id": m_id,
                                "content": item,
                                "timestamp": datetime.datetime.now().isoformat()
                            })
                            max_id = max(max_id, i+1)
                        elif isinstance(item, dict):
                            if "id" not in item:
                                m_id = f"M{i+1}"
                                item["id"] = m_id
                            else:
                                m_id = item["id"]
                                
                            try:
                                max_id = max(max_id, int(m_id[1:]))
                            except ValueError:
                                pass
                                
                            if "timestamp" not in item:
                                item["timestamp"] = datetime.datetime.now().isoformat()
                                
                            upgraded_memories.append(item)
                            
                    self.m_counter = max_id
                    return upgraded_memories
            except Exception:
                return []
        return []

    def save_memories(self):
        try:
            with open(MEMORIES_PATH, "w", encoding="utf-8") as f:
                json.dump(self.memories, f, indent=4)
        except Exception as e:
            self.print_color("system", f"Failed to save memories: {e}")

    def save_conversation(self, filename):
        path = os.path.join(SESSIONS_DIR, filename)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=4)
        except Exception as e:
            self.print_color("system", f"Error saving matrix tracks: {e}")

    def load_conversation(self, filename):
        path = os.path.join(SESSIONS_DIR, filename)
        if not os.path.exists(path):
            self.print_color("system", f"Track asset file '{filename}' was not located.\n")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    if not data or data[0].get("role") != "system":
                        data.insert(0, self.system_prompt)
                    self.history = self._upgrade_history(data)
                    self.current_file = filename
                    self.print_color("system", f"Successfully restored data path state: {filename}\n")
                    
                    self.print_color("system", "--- Restored Conversation ---")
                    for msg in self.history[1:]:
                        role = msg.get("role")
                        content = msg.get("content", "")
                        tag = msg.get("id", "")
                        if role == "user":
                            self.print_color("user", f"{self.config.get('user_name')} [{tag}] > {content}")
                        elif role == "assistant":
                            self.print_color("assistant", f"{self.config.get('assistant_name')} [{tag}] > {content}")
                    print()
                else:
                    self.print_color("system", "Error: Save asset format is corrupted.\n")
        except Exception as e:
            self.print_color("system", f"Loading failure: {e}\n")

    def get_truncated_context(self, content, term, num_words=50):
        term_lower = term.lower()
        content_lower = content.lower()
        idx = content_lower.find(term_lower)
        if idx == -1:
            return content
        before_text = content[:idx]
        after_text = content[idx + len(term):]
        before_words = before_text.split()
        after_words = after_text.split()
        start_ellipsis = "... " if len(before_words) > num_words else ""
        end_ellipsis = " ..." if len(after_words) > num_words else ""
        before_snippet = " ".join(before_words[-num_words:])
        after_snippet = " ".join(after_words[:num_words])
        original_term = content[idx:idx + len(term)]
        parts = []
        if before_snippet: parts.append(start_ellipsis + before_snippet)
        parts.append(original_term)
        if after_snippet: parts.append(after_snippet + end_ellipsis)
        return " ".join(parts).strip()

    def build_system_prompt(self):
        bot_name = self.config.get("assistant_name")
        mode = self.config.get("detail_mode", "medium")
        
        if mode == "low":
            detail_instruction = "Use the absolute MINIMUM number of output tokens. Be hyper-concise. Output the shortest possible answer. USE UNICODE ONLY."
        elif mode == "high":
            detail_instruction = "Provide highly detailed, exhaustive explanations. Give the AI full access to explain concepts thoroughly without token constraints. USE UNICODE ONLY."
        else:
            detail_instruction = "Keep your answers medium length. Be clear and helpful, but not overly verbose or hyper-concise. USE UNICODE ONLY."

        return {
            "id": "SYS",
            "role": "system",
            "content": (
                f"You are {bot_name}, the user's loyal AI assistant. "
                f"Your name is strictly {bot_name}. {detail_instruction} "
                "CRITICAL: Standard terminals cannot parse raw LaTeX. You MUST NEVER use LaTeX markup or math block delimiters like $$, $, \\[, \\], \\(, or \\). You MUST use explicit Unicode math characters (e.g., √, ², π, θ, ∫) and basic plain text formatting instead. "
                "When referencing previous conversation turns, you will see them labeled as [O1], [A1], etc. Treat these as continuous conversational context."
            )
        }

    def get_color(self, entity):
        return COLORS.get(self.config.get(f"{entity}_color"), COLORS["reset"])

    def print_color(self, entity, text, end="\n", flush=False):
        print(f"{self.get_color(entity)}{text}{COLORS['reset']}", end=end, flush=flush)

    def print_timestamp(self, entity):
        if self.config.get("timestamps"):
            ts = datetime.datetime.now().strftime("[%H:%M:%S]")
            print(f"{self.get_color('system')}{ts}{COLORS['reset']} ", end="", flush=True)

    def display_header(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        bot_name = str(self.config.get('assistant_name')).upper()
        self.print_color("system", "=" * 55)
        self.print_color("system", f"    {bot_name} TERMINAL INTERFACE - HYBRID MEMORY MATRIX")
        self.print_color("system", "=" * 55)
        print()
        self.print_color("system", f"Active Provider: {self.config.get('provider').upper()}")
        self.print_color("system", f"Active Model   : {self.config.get('model')}")
        self.print_color("system", f"Memory Bank    : {len(self.memories)} core facts stored")
        print()
        self.print_color("system", "Type /help for commands. Use trailing '\\' for multiline.")
        print()

    def process_command(self, user_input):
        parts = user_input.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd == "/help":
            self.print_color("system", "\nCommands:")
            self.print_color("system", "  /provider              - List and setup API providers interactively")
            self.print_color("system", "  /profile               - Manage saved provider and model configurations")
            self.print_color("system", "  /models                - Fetch all available models from the active provider")
            self.print_color("system", "  /model <name>          - Change the model string for the active provider")
            self.print_color("system", "  /name <user|bot> <name>- Change your name or the bot's name")
            self.print_color("system", "  /switch <prov> <mod>   - Switch provider and model simultaneously")
            self.print_color("system", "  /save <name>           - Save the current conversation")
            self.print_color("system", "  /load <name>           - Load a conversation file")
            self.print_color("system", "  /delete <name>         - Delete a conversation file")
            self.print_color("system", "  /list                  - List all saved conversations")
            self.print_color("system", "  /search <word>         - Scan chats (supports 'chat <name>', 'n:<num>', quotes)")
            self.print_color("system", "  /remember <text>       - Summarize, deduplicate, and store a fact in RAG")
            self.print_color("system", "  /memories              - View all saved long-term memories and timestamps")
            self.print_color("system", "  /forget <id>           - Delete a specific memory (e.g., /forget M1)")
            self.print_color("system", "  /clear                 - Wipe active conversation memory")
            self.print_color("system", "  /detail <level>        - Set response detail level: low, medium, high")
            self.print_color("system", "  /cap <number>          - Set an input token cap (or 'uncapped')")
            self.print_color("system", "  /config                - View/Edit system configurations")
            self.print_color("system", "  /exit                  - Close application\n")
            
        elif cmd == "/provider":
            providers = self.config.get("providers")
            if not args:
                self.print_color("system", f"\nCurrent Active Provider: {self.config.get('provider')}")
                self.print_color("system", "Available Configurations:")
                for p_name, data in providers.items():
                    self.print_color("system", f"  - {p_name} -> URL: {data['url']} | Env: {data['env_var']}")
                print()
                self.print_color("system", "Sub-commands:")
                self.print_color("system", "  /provider switch <name>  - Swap to active provider")
                self.print_color("system", "  /provider add            - Start interactive setup dialog")
                self.print_color("system", "  /provider remove <name>  - Delete a saved provider")
                print()
            elif args[0].lower() == "switch" and len(args) == 2:
                target = args[1].lower()
                if target in providers:
                    self.config.set("provider", target)
                    self.client.update_provider(target)
                    self.display_header()
                else:
                    self.print_color("system", f"Unknown provider block: {target}")
            elif args[0].lower() == "add":
                self.print_color("system", "\n--- Provider Interactive Configuration Setup ---")
                try:
                    p_name = input(f"{self.get_color('system')}1. Enter Provider Nickname (e.g., openrouter, deepseek): {COLORS['reset']}").strip().lower()
                    if not p_name: raise ValueError("Name cannot be empty.")
                    
                    p_url = input(f"{self.get_color('system')}2. Paste Provider Base API URL: {COLORS['reset']}").strip()
                    if not p_url: raise ValueError("URL cannot be empty.")
                    
                    p_env = input(f"{self.get_color('system')}3. Enter Local Environment Variable Name for the key: {COLORS['reset']}").strip()
                    if not p_env: raise ValueError("Env name cannot be empty.")
                    
                    providers[p_name] = {"url": p_url, "env_var": p_env}
                    self.config.set("providers", providers)
                    self.print_color("system", f"\nSuccessfully configured provider matrix for '{p_name}'!\n")
                except (KeyboardInterrupt, ValueError) as e:
                    self.print_color("system", f"\nSetup canceled or invalid: {e}\n")
            elif args[0].lower() in ["remove", "delete"] and len(args) == 2:
                target = args[1].lower()
                if target in providers:
                    if target == self.config.get("provider"):
                        self.print_color("system", f"Cannot delete the currently active provider. Switch first.\n")
                    else:
                        del providers[target]
                        self.config.set("providers", providers)
                        self.print_color("system", f"Provider '{target}' successfully deleted.\n")
                else:
                    self.print_color("system", f"Unknown provider: {target}\n")

        elif cmd == "/models":
            active_provider = self.config.get("provider").upper()
            self.print_color("system", f"\nQuerying {active_provider} servers for available models...")
            models_list = self.client.get_models()
            for m in models_list:
                self.print_color("system", f"  - {m}")
            print()
        
        elif cmd == "/switch":
            if len(args) < 2:
                self.print_color("system", "Usage: /switch <provider_name> <model_string>")
            else:
                target_provider, target_model = args[0].lower(), args[1]
                if target_provider in self.config.get("providers"):
                    self.config.set("provider", target_provider)
                    self.config.set("model", target_model)
                    self.client.update_provider(target_provider)
                    self.display_header()
                else:
                    self.print_color("system", f"Provider '{target_provider}' is not in config. Register it via /provider add.")

        elif cmd == "/profile":
            profiles = self.config.get("profiles")
            if not profiles and not isinstance(profiles, dict):
                profiles = {}
                
            if not args:
                self.print_color("system", "\n--- Saved Profiles ---")
                if not profiles:
                    self.print_color("system", "  No profiles saved yet.")
                else:
                    for p_name, p_data in profiles.items():
                        self.print_color("system", f"  - {p_name} -> Provider: {p_data['provider']} | Model: {p_data['model']}")
                print()
                self.print_color("system", "Sub-commands:")
                self.print_color("system", "  /profile save <name>    - Save current provider & model")
                self.print_color("system", "  /profile switch <name>  - Switch to a saved profile")
                self.print_color("system", "  /profile delete <name>  - Delete a saved profile")
                print()
            elif args[0].lower() == "save" and len(args) == 2:
                p_name = args[1].lower()
                profiles[p_name] = {
                    "provider": self.config.get("provider"),
                    "model": self.config.get("model")
                }
                self.config.set("profiles", profiles)
                self.print_color("system", f"Profile '{p_name}' successfully saved!\n")
            elif args[0].lower() == "switch" and len(args) == 2:
                p_name = args[1].lower()
                if p_name in profiles:
                    p_data = profiles[p_name]
                    target_provider = p_data["provider"]
                    if target_provider in self.config.get("providers"):
                        self.config.set("provider", target_provider)
                        self.config.set("model", p_data["model"])
                        self.client.update_provider(target_provider)
                        self.display_header()
                    else:
                        self.print_color("system", f"Error: Provider '{target_provider}' in this profile no longer exists.\n")
                else:
                    self.print_color("system", f"Unknown profile: '{p_name}'\n")
            elif args[0].lower() in ["delete", "remove"] and len(args) == 2:
                p_name = args[1].lower()
                if p_name in profiles:
                    del profiles[p_name]
                    self.config.set("profiles", profiles)
                    self.print_color("system", f"Profile '{p_name}' deleted.\n")
                else:
                    self.print_color("system", f"Unknown profile: '{p_name}'\n")

        elif cmd == "/model":
            if not args:
                self.print_color("system", f"Current model string: {self.config.get('model')}")
            else:
                self.config.set("model", args[0])
                self.display_header()
        
        elif cmd == "/name":
            if len(args) < 2 or args[0].lower() not in ["user", "bot"]:
                self.print_color("system", "Usage: /name <user|bot> <new_name>\n")
            else:
                target = args[0].lower()
                new_name = " ".join(args[1:])
                if target == "user":
                    self.config.set("user_name", new_name)
                    self.print_color("system", f"User name changed to: {new_name}\n")
                elif target == "bot":
                    self.config.set("assistant_name", new_name)
                    self.system_prompt = self.build_system_prompt()
                    if self.history and self.history[0].get("role") == "system":
                        self.history[0] = self.system_prompt
                    self.print_color("system", f"Bot name changed to: {new_name}\n")
                    self.display_header()
        
        elif cmd == "/detail":
            if not args or args[0].lower() not in ["low", "medium", "high"]:
                self.print_color("system", f"Current detail level: {self.config.get('detail_mode', 'medium')}")
                self.print_color("system", "Usage: /detail <low|medium|high>\n")
            else:
                level = args[0].lower()
                self.config.set("detail_mode", level)
                self.system_prompt = self.build_system_prompt()
                if self.history and self.history[0].get("role") == "system":
                    self.history[0] = self.system_prompt
                self.print_color("system", f"Detail level set to: {level}\n")
                
        elif cmd == "/cap":
            if not args:
                cap = self.config.get("token_cap", 0)
                status = "uncapped" if not cap else str(cap)
                self.print_color("system", f"Current input token cap: {status}")
                self.print_color("system", "Usage: /cap <number|uncapped>\n")
            else:
                val = args[0].lower()
                if val in ["uncapped", "none", "0"]:
                    self.config.set("token_cap", 0)
                    self.print_color("system", "Input token cap disabled (uncapped).\n")
                else:
                    try:
                        cap = int(val)
                        if cap < 50:
                            self.print_color("system", "Cap too low! Minimum is 50 tokens.\n")
                        else:
                            self.config.set("token_cap", cap)
                            self.print_color("system", f"Input token cap set to: {cap} tokens.\n")
                    except ValueError:
                        self.print_color("system", "Invalid value. Must be a number or 'uncapped'.\n")
                
        elif cmd == "/config":
            if not args:
                self.print_color("system", "\n--- Global Configurations ---")
                for key, val in self.config.config.items():
                    if key != "providers":
                        self.print_color("system", f"  {key}: {val}")
                print()
            elif len(args) < 2:
                self.print_color("system", "Usage: /config <key> <value>")
            else:
                key, val = args[0], " ".join(args[1:])
                success, message = self.config.set(key, val)
                self.print_color("system", message + "\n")
                if success and key == "assistant_name":
                    self.system_prompt = self.build_system_prompt()
                    if self.history and self.history[0].get("role") == "system":
                        self.history[0] = self.system_prompt
                    self.display_header()

        elif cmd == "/save":
            if not args:
                self.print_color("system", "Usage: /save <filename>")
            else:
                # strip directory/path traversal attempts (eg., ../../file.txt) by using basename
                safe_name = os.path.basename(args[0])
                name = safe_name if safe_name.endswith(".txt") else f"{safe_name}.txt"
                self.current_file = name
                self.save_conversation(name)
                self.print_color("system", f"Conversation Matrix dumped safely to: {name}\n")

        elif cmd == "/load":
            if not args:
                self.print_color("system", "Usage: /load <filename>")
            else:
                # strip directory/path traversal attempts (eg., ../../file.txt) by using basename
                safe_name = os.path.basename(args[0])
                name = safe_name if safe_name.endswith(".txt") else f"{safe_name}.txt"
                self.load_conversation(name)

        elif cmd == "/list":
            files = [f for f in os.listdir(SESSIONS_DIR) if f.endswith(".txt") and f != DEFAULT_AUTOSAVE_FILE]
            if not files:
                self.print_color("system", "No saved states found.\n")
            else:
                self.print_color("system", "\nSaved Context Tracks:")
                for f in sorted(files):
                    self.print_color("system", f"  - {f.replace('.txt', '')}")
                print()

        elif cmd == "/clear":
            self.history = [self.system_prompt]
            self.o_counter = 0
            self.a_counter = 0
            self.print_color("system", "Memory array wiped clean.\n")

        elif cmd == "/remember":
            if not args:
                self.print_color("system", "Usage: /remember <fact or information>")
                return False
                
            raw_text = " ".join(args)
            self.print_color("system", " [Librarian analyzing memory core...]", end="")
            sys.stdout.flush()
            
            summary = ""
            try:
                response = requests.post("http://127.0.0.1:11434/api/generate", json={
                    "model": "qwen2.5:3b",
                    "prompt": f"Extract the core fact from this text into a single, ultra-concise sentence. Write it as a strict database fact about 'The user' (e.g., 'The user\\'s name is X' or 'The user likes Y'). Do not add conversational filler. Text: {raw_text}",
                    "stream": False,
                    "keep_alive": -1
                }, timeout=60)
                response.raise_for_status()
                summary = response.json().get("response", "").strip()
            except Exception as e:
                self.print_color("red", f"\n[Warning: Memory extraction failed ({type(e).__name__}). Saving raw text instead.]")
                summary = f"[UNVERIFIED RAW MEMORY] {raw_text}"
                
            duplicate_id = "NONE"
            if self.memories:
                memory_map = "\n".join([f"[{m['id']}]: {m['content']}" for m in self.memories])
                dedup_prompt = (
                    "You are a strict memory deduplication engine.\n"
                    "Analyze the NEW MEMORY against the EXISTING MEMORIES.\n\n"
                    "Output ONLY valid JSON with the following schema:\n"
                    "{\n"
                    '  "action": "NONE" | "UPDATE" | "REDUNDANT" | "MERGE",\n'
                    '  "target_id": "M<id>" | null,\n'
                    '  "merged_text": "merged sentence here" | null\n'
                    "}\n\n"
                    "Rules for 'action':\n"
                    "- REDUNDANT: The new memory provides NO NEW INFORMATION because it is already covered by an existing memory (e.g. subset, or exact duplicate).\n"
                    "- UPDATE: The new memory changes a specific fact in an existing memory (e.g. changing age from 15 to 16, or updating a preference).\n"
                    "- MERGE: The new memory adds related information to an existing topic (e.g. likes Physics + likes Chemistry = likes Physics and Chemistry). Provide the combined sentence in 'merged_text'.\n"
                    "- NONE: The new memory is a completely new topic (e.g. Name vs Age).\n\n"
                    "Examples:\n"
                    "Existing Memories:\n[M1] User likes robotics, AI and Arduino.\n"
                    "New Memory:\nUser likes AI.\n"
                    'Output:\n{"action": "REDUNDANT", "target_id": "M1", "merged_text": null}\n\n'
                    "Existing Memories:\n[M1] User likes Physics.\n"
                    "New Memory:\nUser likes Chemistry.\n"
                    'Output:\n{"action": "MERGE", "target_id": "M1", "merged_text": "The user likes Physics and Chemistry."}\n\n'
                    "Existing Memories:\n[M1] User lives in New York.\n"
                    "New Memory:\nUser resides in New York.\n"
                    'Output:\n{"action": "REDUNDANT", "target_id": "M1", "merged_text": null}\n\n'
                    "Existing Memories:\n[M1] User's name is Pranshoo.\n"
                    "New Memory:\nUser's age is 16.\n"
                    'Output:\n{"action": "NONE", "target_id": null, "merged_text": null}\n\n'
                    f"Existing Memories:\n{memory_map}\n\n"
                    f"New Memory:\n{summary}\n\n"
                    "Output:"
                )
                try:
                    res = requests.post("http://127.0.0.1:11434/api/generate", json={"model": "qwen2.5:3b", "prompt": dedup_prompt, "stream": False, "format": "json", "keep_alive": -1}, timeout=60)
                    resp_json = res.json().get("response", "").strip()
                    parsed = json.loads(resp_json)
                    action = parsed.get("action", "NONE")
                    target_id = parsed.get("target_id")
                    merged_text = parsed.get("merged_text")
                    
                    if action == "REDUNDANT" and target_id and target_id.startswith("M"):
                        self.print_color("system", f"Fact already known in {target_id}. Ignoring to prevent redundancy.")
                        sys.stdout.write("\r\033[K")
                        return False
                    elif action == "MERGE" and target_id and target_id.startswith("M") and merged_text:
                        duplicate_id = target_id
                        summary = merged_text
                        self.print_color("yellow", f"Merge detected with {target_id}. Proposing combined memory.")
                    elif action == "UPDATE" and target_id and target_id.startswith("M"):
                        duplicate_id = target_id
                    else:
                        duplicate_id = "NONE"
                except Exception as e:
                    self.print_color("red", f"\n[Deduplication Error: {e}]")
                    duplicate_id = "NONE"
                    
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            
            if duplicate_id != "NONE":
                target_mem = next((m for m in self.memories if m["id"] == duplicate_id), None)
                if target_mem:
                    self.print_color("yellow", f"Collision detected with {duplicate_id}: '{target_mem['content']}'")
                    choice = input(f"{self.get_color('system')}Update {duplicate_id} with new info? (y/N) > {COLORS['reset']}").strip().lower()
                    if choice == 'y':
                        target_mem['content'] = summary
                        target_mem['timestamp'] = datetime.datetime.now().isoformat()
                        self.save_memories()
                        self.print_color("system", f"{duplicate_id} successfully overwritten.\n")
                    else:
                        self.print_color("system", "Memory update aborted.\n")
                    return False
            
            self.m_counter += 1
            new_id = f"M{self.m_counter}"
            self.memories.append({
                "id": new_id,
                "content": summary, 
                "timestamp": datetime.datetime.now().isoformat()
            })
            self.save_memories()
            self.print_color("system", f"Stored in Memory Core [{new_id}]: {summary}\n")

        elif cmd == "/memories":
            if not self.memories:
                self.print_color("system", "Memory Core is currently empty.\n")
            else:
                self.print_color("system", "\n--- Active Long-Term Memories ---")
                for mem in self.memories:
                    try:
                        dt = datetime.datetime.fromisoformat(mem['timestamp']).strftime('%Y-%m-%d %H:%M')
                    except:
                        dt = "Unknown Time"
                    self.print_color("system", f"  [{mem['id']}] (Updated: {dt})\n      {mem['content']}")
                print()

        elif cmd == "/forget":
            if not args:
                self.print_color("system", "Usage: /forget M<number> [M<number>...] (e.g., /forget M1 M2 M3)")
            else:
                forgotten = []
                not_found = []
                for arg in args:
                    target_id = arg.upper()
                    if not target_id.startswith("M"):
                        not_found.append(target_id)
                        continue
                    initial_len = len(self.memories)
                    self.memories = [m for m in self.memories if m["id"] != target_id]
                    if len(self.memories) < initial_len:
                        forgotten.append(target_id)
                    else:
                        not_found.append(target_id)
                
                if forgotten:
                    # deleted re-indexing loop, because shifting M-tags breaks old chat logs that reference to specific ID
                    self.save_memories()
                    self.print_color("system", f"Erased from memory: {', '.join(forgotten)}\n")
                if not_found:
                    self.print_color("system", f"Not found or invalid: {', '.join(not_found)}\n")

        elif cmd == "/delete":
            if not args:
                self.print_color("system", "Usage: /delete <chatname>")
            else:
                # strip directory/path traversal attempts (eg., ../../file.txt) by using basename
                safe_name = os.path.basename(args[0])
                filename = safe_name if safe_name.endswith(".txt") else f"{safe_name}.txt"
                path = os.path.join(SESSIONS_DIR, filename)
                
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        self.print_color("system", f"Deleted chat {filename}.\n")
                        if self.current_file == filename:
                            self.current_file = DEFAULT_AUTOSAVE_FILE
                            self.history = [self.system_prompt]
                            self.o_counter = 0
                            self.a_counter = 0
                            self.print_color("system", "Active chat was deleted. Starting a new session.\n")
                    except Exception as e:
                        self.print_color("system", f"Failed to delete {filename}: {e}\n")
                else:
                    self.print_color("system", f"Chat {filename} not found.\n")

        elif cmd == "/search":
            search_args_str = " ".join(args).strip()
            if not search_args_str:
                self.print_color("system", "Usage: /search <word> [n:<limit>] OR /search chat <filename> <word>")
            else:
                limit = 5
                n_match = re.search(r'\bn:(\d+)\b', search_args_str)
                if n_match:
                    limit = int(n_match.group(1))
                    search_args_str = search_args_str[:n_match.start()] + search_args_str[n_match.end():]
                    search_args_str = search_args_str.strip()
                    
                target_files = []
                term = ""
                match = re.match(r'(?i)^(?:chat|in)\s+(\S+)\s+(.+)$', search_args_str)
                if match:
                    filename = match.group(1)
                    if not filename.endswith(".txt"): filename += ".txt"
                    target_files = [filename]
                    term = match.group(2)
                else:
                    target_files = [f for f in os.listdir(SESSIONS_DIR) if f.endswith(".txt") and f != DEFAULT_AUTOSAVE_FILE]
                    term = search_args_str
                    
                if (term.startswith('"') and term.endswith('"')) or (term.startswith("'") and term.endswith("'")):
                    term = term[1:-1]

                self.print_color("system", f"\nScanning memory matrix for: '{term}' (limit: {limit}/file)...")
                term_lower = term.lower()
                results = {}
                
                for file in target_files:
                    path = os.path.join(SESSIONS_DIR, file)
                    if not os.path.exists(path): continue
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if not isinstance(data, list): continue
                        
                        temp_o, temp_a = 0, 0
                        for msg in data:
                            if msg.get("role") == "system": continue
                            
                            if "id" in msg:
                                tag = msg["id"]
                            else:
                                if msg.get("role") == "user":
                                    temp_o += 1; tag = f"O{temp_o}"
                                else:
                                    temp_a += 1; tag = f"A{temp_a}"
                                    
                            content = msg.get("content", "")
                            
                            if term_lower in content.lower():
                                if file not in results:
                                    results[file] = []
                                if len(results[file]) >= limit:
                                    break
                                trunc_content = self.get_truncated_context(content, term, 50)
                                results[file].append((tag, trunc_content))
                                
                    except json.JSONDecodeError:
                        pass
                        
                if not results:
                    self.print_color("system", "No matches found in the target sector.\n")
                else:
                    for file, matches in results.items():
                        clean_name = file.replace('.txt', '')
                        self.print_color("system", f"\n{clean_name}.txt")
                        self.print_color("system", "-" * len(clean_name))
                        for tag, content in matches:
                            self.print_color("cyan", f"[{tag}]")
                            print(f"{content}\n")

        elif cmd == "/exit":
            user = self.config.get("user_name")
            self.print_color("system", f"Disconnecting shell matrix. Goodbye, {user}!")
            return True
        else:
            self.print_color("system", "Unrecognized routing script instruction. Use /help.\n")
        return False

    def get_librarian_context(self, current_user_input):
        history_map = ""
        for msg in self.history[1:]:
            tag = msg.get("id", "SYS")
            content_snippet = msg['content'].replace('\n', ' ')
            history_map += f"[{tag}]: {content_snippet}\n"

        memory_map = ""
        for mem in self.memories:
            memory_map += f"[{mem['id']}]: {mem['content']}\n"

        if not history_map.strip() and not memory_map.strip():
            return [self.system_prompt]

        librarian_instruction = (
            f"You are a strict context extractor. Your ONLY job is to search the provided History Map and Memory Bank.\n\n"
            f"User Query: '{current_user_input}'\n\n"
            f"Active Chat Model: {self.config.get('provider')}/{self.config.get('model')}\n\n"
            f"History Map:\n{history_map}\n\n"
            f"Memory Bank:\n{memory_map}\n\n"
            f"Instructions:\n"
            f"1. Extract and return ONLY the actual text from the History Map or Memory Bank that is relevant to the User Query.\n"
            f"2. DO NOT answer the user's query yourself. NEVER generate external knowledge.\n"
            f"3. If the provided History Map and Memory Bank do not contain relevant information to answer the query, you MUST reply with exactly 'NONE'."
        )

        payload = {
            "model": "qwen2.5:3b", 
            "prompt": librarian_instruction, 
            "stream": False, 
            "keep_alive": -1,
            "options": {
                "temperature": 0.0,
                "num_predict": 250
            }
        }

        try:
            self.print_color("system", " [Librarian indexing local context & memories...]", end="")
            sys.stdout.flush()
            
            response = requests.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=60)
            response.raise_for_status()
            
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            
            extracted_context = response.json().get("response", "").strip()
            
            optimized_payload = [self.system_prompt]
            
            active_provider = self.config.get("provider")
            active_model = self.config.get("model")
            optimized_payload.append({
                "role": "system",
                "content": f"[SYSTEM INFO]: The active chat provider is {active_provider} and the model is {active_model}."
            })
            
            if extracted_context and extracted_context.upper() != "NONE":
                optimized_payload.append({
                    "role": "system",
                    "content": f"[LIBRARIAN EXTRACTED CONTEXT]: {extracted_context}"
                })
            
            for mem in self.memories:
                if mem["id"] in current_user_input.upper():
                    optimized_payload.append({
                        "role": "system",
                        "content": f"[SYSTEM RECALLED MEMORY {mem['id']}]: {mem['content']}"
                    })

            # Inject Selected History
            recent_msgs = self.history[-2:] if len(self.history) > 2 else self.history[1:]
            for msg in self.history[1:]:
                current_tag = msg.get("id", "")
                is_recent = msg in recent_msgs
                
                if current_tag in current_user_input.upper():
                    if is_recent:
                        optimized_payload.append({"role": msg["role"], "content": f"[{current_tag}]: {msg['content']}"})
                    else:
                        try:
                            self.print_color("system", f"\r\033[K [Librarian summarizing historical {current_tag}...]", end="")
                            sys.stdout.flush()
                            sum_payload = {
                                "model": "qwen2.5:3b",
                                "prompt": f"Summarize the core points of this past message as concisely as possible:\n\n{msg['content']}",
                                "stream": False,
                                "keep_alive": -1,
                                "options": {"temperature": 0.0, "num_predict": 100}
                            }
                            sum_response = requests.post("http://127.0.0.1:11434/api/generate", json=sum_payload, timeout=30)
                            sum_response.raise_for_status()
                            summarized_text = sum_response.json().get("response", "").strip()
                            optimized_payload.append({"role": msg["role"], "content": f"[{current_tag} - SUMMARIZED]: {summarized_text}"})
                            self.print_color("system", f"\r\033[K [Librarian indexing local context & memories...]", end="")
                            sys.stdout.flush()
                        except Exception as e:
                            self.print_color("red", f"\r\033[K[Failed to summarize {current_tag}, injecting raw]")
                            sys.stdout.flush()
                            optimized_payload.append({"role": msg["role"], "content": f"[{current_tag}]: {msg['content']}"})
                elif is_recent:
                    optimized_payload.append({"role": msg["role"], "content": f"[{current_tag}]: {msg['content']}"})
            
            return optimized_payload

        except Exception as e:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            self.print_color("red", f"[Warning: Librarian failed to index context ({type(e).__name__}). Falling back to recent history.]\n")
            active_provider = self.config.get("provider")
            active_model = self.config.get("model")
            return [
                self.system_prompt,
                {"role": "system", "content": f"[SYSTEM INFO]: The active chat provider is {active_provider} and the model is {active_model}."}
            ] + self.history[-2:]

    def process_ai_response(self, raw_user_input):
        model = self.config.get("model")
        
        self.o_counter += 1
        o_tag = f"O{self.o_counter}"
        
        optimized_context = self.get_librarian_context(raw_user_input)
        
        self.history.append({"id": o_tag, "role": "user", "content": raw_user_input})
        optimized_context.append({"role": "user", "content": f"[{o_tag}]: {raw_user_input}"})
        
        # Clean the ID tags out before sending to strict provider APIs
        clean_context = [{"role": m["role"], "content": m["content"]} for m in optimized_context]
        
        token_cap = self.config.get("token_cap", 0)
        if token_cap > 0:
            def est_tokens(ctx):
                return sum(len(m["content"]) // 4 for m in ctx)
            
            while est_tokens(clean_context) > token_cap and len(clean_context) > 2:
                removed = False
                for i in range(1, len(clean_context) - 1):
                    if clean_context[i]["role"] != "system":
                        clean_context.pop(i)
                        removed = True
                        break
                if not removed:
                    break
        
        generator = self.client.stream_completion(model, clean_context)
        bot_name = self.config.get("assistant_name")
        
        self.a_counter += 1
        a_tag = f"A{self.a_counter}"
        
        self.print_timestamp("assistant")
        self.print_color("assistant", f"{bot_name} [{a_tag}] > ", end="")
        
        full_response = ""
        buffer = ""
        in_think_block = False
        assistant_color = self.get_color("assistant")
        reset_color = COLORS["reset"]
        
        print(assistant_color, end="", flush=True)
        
        try:
            for chunk in generator:
                buffer += chunk
                if not in_think_block:
                    if "<think>" in buffer:
                        parts = buffer.split("<think>", 1)
                        print(parts[0], end="", flush=True)
                        full_response += parts[0]
                        buffer = parts[1]
                        in_think_block = True
                    else:
                        partial = False
                        for i in range(1, len("<think>")):
                            if buffer.endswith("<think>"[:i]):
                                partial = True
                                print(buffer[:-i], end="", flush=True)
                                full_response += buffer[:-i]
                                buffer = buffer[-i:]
                                break
                        if not partial:
                            print(buffer, end="", flush=True)
                            full_response += buffer
                            buffer = ""

                if in_think_block:
                    if "</think>" in buffer:
                        buffer = buffer.split("</think>", 1)[1]
                        in_think_block = False
                    else:
                        partial = False
                        for i in range(1, len("</think>")):
                            if buffer.endswith("</think>"[:i]):
                                partial = True
                                buffer = buffer[-i:]
                                break
                        if not partial: buffer = ""
                            
            if buffer and not in_think_block:
                print(buffer, end="", flush=True)
                full_response += buffer
                
        except KeyboardInterrupt:
            self.print_color("system", "\n[Streaming Pipeline Aborted By User]")
        finally:
            print(reset_color)
            print()
            if full_response.strip():
                self.history.append({"id": a_tag, "role": "assistant", "content": full_response.strip()})
                if self.config.get("autosave"):
                    self.save_conversation(self.current_file)

    def get_multiline_input(self):
        user_color = self.get_color("user")
        reset_color = COLORS["reset"]
        user_name = self.config.get("user_name")
        
        turn_counter = self.o_counter + 1
        self.print_timestamp("user")
        prompt_str = f"{user_color}{user_name} [O{turn_counter}] > {reset_color}"
        
        lines = []
        while True:
            try:
                line = input(prompt_str)
                if line.endswith("\\"):
                    lines.append(line[:-1])
                    prompt_str = f"{user_color}... > {reset_color}"
                else:
                    lines.append(line)
                    break
            except (KeyboardInterrupt, EOFError):
                raise
        return "\n".join(lines).strip()

    def run(self):
        self.display_header()
        while True:
            try:
                user_input = self.get_multiline_input()
            except KeyboardInterrupt:
                self.print_color("system", "\nUse /exit or Ctrl+D to sever connection execution.")
                continue
            except EOFError:
                user = self.config.get("user_name")
                self.print_color("system", f"\nDisconnecting shell matrix. Goodbye, {user}!")
                break
                
            if not user_input: continue
            if user_input.startswith("/"):
                if self.process_command(user_input): break
                continue
            self.process_ai_response(user_input)

if __name__ == "__main__":
    app = ChatApplication()
    app.run()
