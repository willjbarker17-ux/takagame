#!/usr/bin/env python3
"""
Wyscout Match ID Collector

Opens browser, captures match IDs as you click through matches.
Also captures download URLs if you go through the export flow.

Usage:
    python wyscout_match_collector.py
"""

import json
import re
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    match_ids = set()
    download_urls = []
    session_data = {"cookies": {}, "access_token": None}

    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    WYSCOUT MATCH ID COLLECTOR                              ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  1. Login to Wyscout                                                      ║
║  2. Go to your league page (e.g., NCAA D1 Sun Belt)                       ║
║  3. Click on each match - IDs will be captured automatically              ║
║  4. (Optional) Click Export → Download to also capture download URLs      ║
║  5. Close browser when done                                               ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        def extract_match_id(url_or_text):
            """Extract match ID from various sources."""
            # Pattern: objId=5787516 or matchId=5787516 or /match/5787516
            patterns = [
                r'objId[=:](\d+)',
                r'matchId[=:](\d+)',
                r'/match/(\d+)',
                r'"objId"\s*:\s*(\d+)',
                r'match[_-]?id[=:](\d+)',
                r'/(\d{6,8})(?:[/_-]|$)',  # 6-8 digit numbers in paths
            ]
            for pattern in patterns:
                match = re.search(pattern, str(url_or_text), re.IGNORECASE)
                if match:
                    return int(match.group(1))
            return None

        def on_request(request):
            url = request.url

            # Capture match IDs from URLs
            match_id = extract_match_id(url)
            if match_id and match_id not in match_ids:
                match_ids.add(match_id)
                print(f"✓ Match ID: {match_id} (from URL)")

            # Capture from POST data
            if request.method == "POST" and "aengine" in url:
                try:
                    body = request.post_data or ""

                    # Look for access_token
                    token_match = re.search(r'"access_token"\s*:\s*"([^"]+)"', body)
                    if token_match:
                        session_data["access_token"] = token_match.group(1)

                    # Look for match ID in request body
                    match_id = extract_match_id(body)
                    if match_id and match_id not in match_ids:
                        match_ids.add(match_id)
                        print(f"✓ Match ID: {match_id} (from API request)")
                except:
                    pass

            # Capture download URLs
            if "cdn" in url and "wyscout" in url and ".mp4" in url:
                if url not in download_urls:
                    download_urls.append(url)
                    print(f"✓ Download URL captured (#{len(download_urls)})")

        def on_response(response):
            url = response.url

            # Check response for match IDs (from list endpoints)
            if "aengine" in url:
                try:
                    text = response.text()
                    # Find all potential match IDs in response
                    ids = re.findall(r'"(?:objId|matchId|match_id)"\s*:\s*(\d+)', text)
                    for mid in ids:
                        mid = int(mid)
                        if mid not in match_ids and mid > 100000:  # Reasonable match ID range
                            match_ids.add(mid)
                            print(f"✓ Match ID: {mid} (from API response)")
                except:
                    pass

            # Capture download URL from response
            if "aengine" in url:
                try:
                    text = response.text()
                    cdn_match = re.search(r"(https://cdn5download\.wyscout\.com/[^'\"]+)", text)
                    if cdn_match:
                        dl_url = cdn_match.group(1)
                        if dl_url not in download_urls:
                            download_urls.append(dl_url)
                            print(f"✓ Download URL captured (#{len(download_urls)})")
                except:
                    pass

        page.on("request", on_request)
        page.on("response", on_response)
        context.on("request", on_request)
        context.on("response", on_response)

        # Handle new pages/popups
        def on_page(new_page):
            new_page.on("request", on_request)
            new_page.on("response", on_response)
            # Check URL of new page
            match_id = extract_match_id(new_page.url)
            if match_id:
                match_ids.add(match_id)
                print(f"✓ Match ID: {match_id} (from new page)")

        context.on("page", on_page)

        # Navigate
        page.goto("https://wyscout.hudl.com")

        print("\n[*] Browser opened. Click on matches to capture IDs.")
        print("[*] Match IDs and URLs are saved automatically.")
        print("[*] Close browser when done.\n")

        # Wait for close
        try:
            page.wait_for_event("close", timeout=0)
        except:
            pass

        # Get cookies
        for cookie in context.cookies():
            session_data["cookies"][cookie["name"]] = cookie["value"]

        browser.close()

    # Save results
    print(f"\n{'='*60}")
    print(f"Captured {len(match_ids)} match IDs")
    print(f"Captured {len(download_urls)} download URLs")

    # Save match IDs
    match_ids_file = Path("match_ids.txt")
    with open(match_ids_file, "w") as f:
        for mid in sorted(match_ids):
            f.write(f"{mid}\n")
    print(f"[+] Match IDs saved to: {match_ids_file}")

    # Save download URLs
    if download_urls:
        urls_file = Path("download_urls.txt")
        with open(urls_file, "w") as f:
            f.write("\n".join(download_urls))
        print(f"[+] Download URLs saved to: {urls_file}")

    # Save session
    session_file = Path("wyscout_session.json")
    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2)
    print(f"[+] Session saved to: {session_file}")

    # Print next steps
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"  1. Get download URLs for all matches:")
    print(f"     python3 scripts/wyscout_api.py --matches {' '.join(str(m) for m in list(match_ids)[:5])} ...")
    print(f"\n  2. Or download captured URLs:")
    print(f"     python3 scripts/wyscout_batch.py --urls download_urls.txt --download")


if __name__ == "__main__":
    main()
