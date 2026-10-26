"""Host-level detection rules and plain-language explanations for the HITDS host sensor.

The sensor (sensor/hitds_host_sensor.py) sends a raw snapshot of the machine: open connections, listening
ports and running process names. The rules live HERE, on the server, so they can be tuned or tested
without redeploying anything to the monitored machine.

What these rules can and cannot do (be honest about it in your report):
  * Name and port lists are trivially evaded: renaming nc.exe or using port 443 defeats them. They catch
    careless or default-configured tools, which is exactly what a lab demo and many real intrusions use.
  * A host sensor sees THIS machine's sockets and processes only. It cannot see an inbound nmap SYN scan
    (nothing is left in the socket table) - that is what the flow-based ML model in HITDS is for.
  * Ports such as 5555 (Android debugging) and 1337 have legitimate uses, so those carry a note and a
    lower severity instead of being called an attack.
Explanations are fixed templates filled with the measured values; no data leaves the server.
"""
from __future__ import annotations

import os

# port -> (severity, why it is notable)
SUSPICIOUS_PORTS = {
    4444: ("HIGH", "the default listener port of the Metasploit penetration-testing framework"),
    31337: ("HIGH", "a classic 'elite' backdoor port"),
    12345: ("MEDIUM", "the default port of old NetBus-style remote-access trojans"),
    1337: ("MEDIUM", "a port often used in hacking tools (and in some games)"),
    5555: ("MEDIUM", "a common reverse-shell port, but also the Android Debug Bridge default"),
}

# process name (lower case, without .exe) -> (severity, what it is)
RESTRICTED_TOOLS = {
    "nc": ("HIGH", "netcat, a tool that can open remote shells"),
    "ncat": ("HIGH", "ncat, a tool that can open remote shells"),
    "netcat": ("HIGH", "netcat, a tool that can open remote shells"),
    "mimikatz": ("HIGH", "mimikatz, a tool that steals passwords from memory"),
    "psexec": ("MEDIUM", "PsExec, which runs commands on other machines (often used for lateral movement)"),
    "psexec64": ("MEDIUM", "PsExec, which runs commands on other machines (often used for lateral movement)"),
    "nmap": ("MEDIUM", "nmap, a network scanner"),
    "masscan": ("MEDIUM", "masscan, a very fast network scanner"),
    "hydra": ("HIGH", "hydra, a password-guessing tool"),
}

FLOOD_CONNECTIONS = int(os.environ.get("HITDS_HOST_FLOOD", "400"))   # a busy desktop is usually 100-300
SCAN_DISTINCT_TARGETS = int(os.environ.get("HITDS_HOST_SCAN", "20"))  # half-open connections to this many targets

LIMITS = {"established": 300, "syn_sent": 500, "listening": 200, "processes": 800}
_LOOPBACK = ("127.", "::1")


def _num(v, lo=0.0, hi=1e9):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return lo if f != f else max(lo, min(f, hi))


def _txt(v, n=64):
    return "" if v is None else str(v)[:n]


def clean_snapshot(body) -> dict:
    """Validate and clamp what a sensor sent. Raises ValueError if it is not a usable snapshot."""
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    snap = {"host": _txt(body.get("host"), 48) or "unknown", "os": _txt(body.get("os"), 24),
            "cpu_percent": _num(body.get("cpu_percent"), 0, 100), "memory_percent": _num(body.get("memory_percent"), 0, 100),
            "total_connections": int(_num(body.get("total_connections"), 0, 1e7))}
    for key, fields in (("established", ("pid", "proc", "laddr", "lport", "raddr", "rport")),
                        ("syn_sent", ("raddr", "rport")), ("listening", ("pid", "proc", "port")),
                        ("processes", ("pid", "name"))):
        rows = body.get(key) or []
        if not isinstance(rows, list):
            raise ValueError(f"{key} must be a list")
        out = []
        for r in rows[:LIMITS[key]]:
            if not isinstance(r, dict):
                continue
            out.append({f: (int(_num(r.get(f), 0, 1e9)) if f in ("pid", "lport", "rport", "port") else _txt(r.get(f)))
                        for f in fields})
        snap[key] = out
    if not any(snap[k] for k in ("established", "listening", "processes")) and not snap["total_connections"]:
        raise ValueError("snapshot contains no measurements")
    return snap


def evaluate(snap: dict) -> list[dict]:
    """Return alerts: {"type", "severity", "key", "details"}. `key` identifies the thing, for de-duplication."""
    alerts = []

    for c in snap.get("established", []):
        info = SUSPICIOUS_PORTS.get(c["rport"])
        if info:
            sev, why = info
            local = c["raddr"].startswith(_LOOPBACK)
            if local:  # same machine on both ends: usually a test, so lower it and say so
                sev = "MEDIUM" if sev == "HIGH" else "LOW"
            alerts.append({"type": "SUSPICIOUS_CONNECTION", "severity": sev,
                           "key": f"{c['proc']}:{c['pid']}:{c['rport']}",
                           "details": f"{c['proc'] or 'a process'} (PID {c['pid']}) is connected to {c['raddr']}:{c['rport']}, "
                                      f"{why}." + (" The other end is this same machine, so this may be a local test." if local else "")})

    for p in snap.get("listening", []):
        info = SUSPICIOUS_PORTS.get(p["port"])
        if info:
            sev, why = info
            alerts.append({"type": "SUSPICIOUS_LISTENER", "severity": sev, "key": f"{p['proc']}:{p['pid']}:{p['port']}",
                           "details": f"{p['proc'] or 'a process'} (PID {p['pid']}) is listening for connections on port {p['port']}, {why}."})

    seen = set()
    for pr in snap.get("processes", []):
        name = pr["name"].lower()
        name = name[:-4] if name.endswith(".exe") else name
        if name in RESTRICTED_TOOLS and name not in seen:
            seen.add(name)
            sev, what = RESTRICTED_TOOLS[name]
            alerts.append({"type": "RESTRICTED_TOOL", "severity": sev, "key": name,
                           "details": f"{pr['name']} (PID {pr['pid']}) is running: {what}."})

    n = snap.get("total_connections", 0)
    if n > FLOOD_CONNECTIONS:
        alerts.append({"type": "SOCKET_FLOOD", "severity": "MEDIUM", "key": "flood",
                       "details": f"{n} open connections (the alert threshold is {FLOOD_CONNECTIONS})."})

    targets = {(r["raddr"], r["rport"]) for r in snap.get("syn_sent", [])}
    if len(targets) >= SCAN_DISTINCT_TARGETS:
        alerts.append({"type": "OUTBOUND_SCAN", "severity": "HIGH", "key": "scan",
                       "details": f"{len(targets)} different destinations are being tried at once with no reply yet, "
                                  "which looks like this machine scanning other hosts (or a worm spreading)."})
    return alerts


# ------------------------------------------------------------------------------------------ plain words
PLAIN = {
    "SUSPICIOUS_CONNECTION": {
        "title": "Connection to a port used by attack tools",
        "what": "A program on this computer opened a connection to a port that attack tools commonly use.",
        "risk": "If it is a reverse shell, someone elsewhere may be able to run commands on this machine.",
        "do": ["Find the program: open Task Manager, Details tab, and look for the PID shown",
               "If you do not recognise it, disconnect from the network and end the process",
               "Check where the remote address is, and run an antivirus scan"]},
    "SUSPICIOUS_LISTENER": {
        "title": "Program waiting for connections on a suspicious port",
        "what": "A program on this computer is listening on a port that backdoors and attack tools commonly use.",
        "risk": "Someone who can reach this machine may be able to connect straight in.",
        "do": ["Find the program using the PID shown and decide whether you started it",
               "If not, stop it and check your startup programs",
               "Make sure the Windows firewall is on and blocks inbound connections"]},
    "RESTRICTED_TOOL": {
        "title": "Security or hacking tool is running",
        "what": "A program that is normally used for scanning, password cracking or remote access is running.",
        "risk": "Fine if you started it on purpose. If you did not, an attacker may be using it from this machine.",
        "do": ["Confirm whether you or an administrator started it",
               "If not, end the process and find out how it got here",
               "Change important passwords if you suspect password theft"]},
    "SOCKET_FLOOD": {
        "title": "Unusually many open connections",
        "what": "This computer has far more network connections open than normal.",
        "risk": "Can mean malware spreading or scanning, a download tool gone wrong, or just very many browser tabs.",
        "do": ["Close heavy programs and see if the number drops",
               "Look at Resource Monitor, Network tab, to see which program owns them",
               "Run an antivirus scan if you cannot explain it"]},
    "OUTBOUND_SCAN": {
        "title": "This machine looks like it is scanning other hosts",
        "what": "It is trying many different destinations at once without getting replies, which is how scanners and worms behave.",
        "risk": "The machine may be infected or misused, and the scan can alert or harm other systems.",
        "do": ["Disconnect the machine from the network", "Find the program making the connections",
               "Run a full antivirus scan before reconnecting"]},
}
CAVEAT = ("This is a rule-based check on one machine's connections and process names. It can be wrong "
          "(for example a developer tool using a common port), so a person should confirm before acting. "
          "HITDS records your decision; it does not block anything by itself.")


def explain(alert_type: str, severity: str, details: str) -> dict:
    p = PLAIN.get(alert_type, {"title": alert_type, "what": "The sensor flagged unusual activity.",
                              "risk": "Unknown.", "do": ["Ask an administrator to review it"]})
    return {"title": p["title"], "what": p["what"], "risk": p["risk"], "do": list(p["do"]),
            "severity": severity, "details": details, "caveat": CAVEAT}
