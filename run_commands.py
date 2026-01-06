#!/usr/bin/env python3
import subprocess

with open("/home/k/kamas7/thesis/masters-thesis/commands.txt", "r") as file:
    commands = [line.strip() for line in file if line.strip() and not line.strip().startswith("#")]

for cmd in commands:
    print(f"▶ Running: {cmd}")

    result = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print(f"✗ Command failed: {cmd}")
        print("----- STDOUT -----")
        print(result.stdout)
        print("----- STDERR -----")
        print(result.stderr)
        break