import json
import sys
from collections import Counter

LOG = "data/logs/requests.jsonl"


def load(sources):
    rows = []
    try:
        with open(LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("source") in sources:
                    rows.append(r)
    except FileNotFoundError:
        sys.exit(f"No log file at {LOG}")
    return rows


def main():
    # default: real traffic and eval runs; pass --all to include test lines
    sources = {"api", "eval", "test"} if "--all" in sys.argv else {"api", "eval"}
    rows = load(sources)
    if not rows:
        sys.exit("No matching requests in the log.")

    costed = [r for r in rows if r.get("estimated_cost") is not None]
    routed = [r for r in costed if r.get("route_tier") and r.get("would_cost_pro") is not None]
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]

    print(f"Requests: {len(rows)}  (sources: {', '.join(sorted(sources))})")
    print(f"With cost data: {len(costed)}   Without (old lines or missing tokens): {len(rows) - len(costed)}")
    print(f"Total estimated cost (Flash, paid rates): ${sum(r['estimated_cost'] for r in costed):.6f}")
    if lat:
        print(f"Average latency: {sum(lat) / len(lat) / 1000:.1f} s   Max: {max(lat) / 1000:.1f} s")

    tiers = Counter(r["route_tier"] for r in routed)
    print(f"\nRouter decisions (dry run): flash={tiers.get('flash', 0)}  pro={tiers.get('pro', 0)}")

    if routed:
        all_flash = sum(r["estimated_cost"] for r in routed)
        all_pro = sum(r["would_cost_pro"] for r in routed)
        with_routing = sum(r["would_cost_pro"] if r["route_tier"] == "pro" else r["estimated_cost"] for r in routed)
        print(f"\nCost of {len(routed)} routed requests under each policy:")
        print(f"  all Flash:       ${all_flash:.6f}")
        print(f"  all Pro:         ${all_pro:.6f}")
        print(f"  with routing:    ${with_routing:.6f}")
        if all_pro > 0:
            print(f"  saving vs all Pro: {100 * (all_pro - with_routing) / all_pro:.0f}%")
        print("  (Pro figures reuse Flash token counts, so they are estimates.)")

    by_src = Counter(r.get("source") for r in rows)
    print(f"\nBy source: {dict(by_src)}")

    try:
        with open("data/logs/blocks.jsonl", "r", encoding="utf-8") as f:
            kinds = Counter(json.loads(l)["kind"] for l in f if l.strip())
        print(f"\nSecurity blocks: {dict(kinds) if kinds else 'none'}")
    except FileNotFoundError:
        print("\nSecurity blocks: none logged yet")


if __name__ == "__main__":
    main()