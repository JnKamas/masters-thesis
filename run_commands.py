#!/usr/bin/env python3
import subprocess

with open("masters-thesis/commands.txt", "r") as file:
    commands = [line.strip() for line in file if line.strip() and not line.strip().startswith("#")]

for cmd in commands:
    print(f"▶ Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"✗ Command failed: {cmd}")
        break
