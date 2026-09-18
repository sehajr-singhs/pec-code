"""List running python processes with command lines (process hygiene helper)."""
import json
import subprocess


def main():
    ps_script = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script],
        capture_output=True, text=True, timeout=40,
    )
    txt = out.stdout.strip()
    if not txt:
        print("no python processes found")
        return
    rows = json.loads(txt)
    if isinstance(rows, dict):
        rows = [rows]
    for r in rows:
        cmd = (r.get("CommandLine") or "").replace("\r", " ").replace("\n", " ")
        # keep it short: last path component + experiment args
        short = cmd.split("\\")[-1] if "\\" in cmd else cmd
        print(r.get("ProcessId"), "  ", short[:120])


if __name__ == "__main__":
    main()
