#!/usr/bin/env python
"""
Hot reload development tool for Maverick-MCP.

This script watches for file changes and automatically restarts the server,
providing instant feedback during development.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:
    print("Installing watchdog for file watching...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "watchdog"])
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer


class ReloadHandler(FileSystemEventHandler):
    """Handler that restarts the server on file changes."""

    def __init__(self, command: list[str], debounce_seconds: float = 0.5):
        self.command = command
        self.debounce_seconds = debounce_seconds
        self.last_reload = 0
        self.process: subprocess.Popen[Any] | None = None
        self.start_process()

    def start_process(self):
        """Start the development process."""
        if self.process:
            print("🔄 Stopping previous process...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

        print(f"🚀 Starting: {' '.join(self.command)}")
        self.process = subprocess.Popen(self.command)
        self.last_reload = time.time()

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        # Skip certain files
        path = Path(event.src_path)
        if any(
            pattern in str(path)
            for pattern in [
                "__pycache__",
                ".pyc",
                ".git",
                ".pytest_cache",
                ".log",
                ".db",
                ".sqlite",
            ]
        ):
            return

        # Only reload Python files and config files
        if path.suffix not in [".py", ".toml", ".yaml", ".yml", ".env"]:
            return

        # Debounce rapid changes
        current_time = time.time()
        if current_time - self.last_reload < self.debounce_seconds:
            return

        print(f"\n📝 File changed: {path}")
        self.start_process()

    def cleanup(self):
        """Clean up the running process."""
        if self.process:
            self.process.terminate()
            self.process.wait()


def main():
    """Main entry point for hot reload."""
    import argparse

    parser = argparse.ArgumentParser(description="Hot reload for Maverick-MCP")
    parser.add_argument(
        "--command",
        default="make backend",
        help="Command to run (default: make backend)",
    )
    parser.add_argument(
        "--watch",
        action="append",
        default=["maverick_mcp"],
        help="Directories to watch (can be specified multiple times)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Patterns to exclude from watching",
    )
    args = parser.parse_args()

    # Parse command
    command = args.command.split() if isinstance(args.command, str) else args.command

    # Set up file watcher
    event_handler = ReloadHandler(command)
    observer = Observer()

    # Watch specified directories
    for watch_dir in args.watch:
        if os.path.exists(watch_dir):
            print(f"👀 Watching: {watch_dir}")
            observer.schedule(event_handler, watch_dir, recursive=True)
        else:
            print(f"⚠️  Directory not found: {watch_dir}")

    observer.start()

    print("\n✨ Hot reload active! Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Stopping hot reload...")
        observer.stop()
        event_handler.cleanup()

    observer.join()


if __name__ == "__main__":
    main()
