#!/usr/bin/env python3
"""
Wyscout Direct API Downloader

Uses Wyscout's internal API to get signed download URLs.
Requires a valid session (cookies) from a logged-in browser.

Usage:
    # Get URLs for specific match IDs
    python wyscout_api.py --matches 5787516 5787517 5787518

    # Interactive: login in browser, then fetch URLs
    python wyscout_api.py --interactive

    # Export cookies from browser, then use them
    python wyscout_api.py --cookies cookies.json --matches 5787516
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests


class WyscoutAPI:
    """Interact with Wyscout's internal API."""

    BASE_URL = "https://wyscout.hudl.com"
    API_ENDPOINT = "/app/aengine-service.php"

    def __init__(self, cookies: dict = None, access_token: str = None):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json, text/html, */*",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin": self.BASE_URL,
            "Referer": f"{self.BASE_URL}/",
        })
        if cookies:
            self.session.cookies.update(cookies)
        self.access_token = access_token

    def get_download_url(self, match_id: int, quality: str = "hd") -> Optional[str]:
        """Get signed download URL for a match."""
        payload = {
            "obj": "download",
            "act": "execute",
            "params": json.dumps({
                "obj": "match",
                "objId": match_id,
                "hide": {"target": "video_export_dialog"},
                "filename": f"match_{match_id}",
                "privacyLevel": 0,
                "mediaType": 1,
                "mediaVersion": 1,
                "hasPlayerSpotlights": False,
                "hasCustomAnalysis": False,
                "visibility": 0,
                "type": 0,
                "quality": quality,
                "access_token": self.access_token or "",
            }),
            "navi": json.dumps({"component": "ae_updater"}),
        }

        try:
            resp = self.session.post(
                urljoin(self.BASE_URL, self.API_ENDPOINT),
                data=payload,
                timeout=30,
            )
            resp.raise_for_status()

            # Response contains: <script>window.location.href = 'URL'</script>
            match = re.search(r"window\.location\.href\s*=\s*'([^']+)'", resp.text)
            if match:
                return match.group(1)

            # Try JSON response
            try:
                data = resp.json()
                if "url" in data:
                    return data["url"]
            except:
                pass

            print(f"[-] Could not parse response for match {match_id}")
            print(f"    Response: {resp.text[:200]}")
            return None

        except Exception as e:
            print(f"[-] Error fetching match {match_id}: {e}")
            return None

    def get_match_list(self, competition_id: int) -> list:
        """Get list of matches for a competition."""
        payload = {
            "obj": "matches",
            "act": "list",
            "params": json.dumps({
                "competitionId": competition_id,
                "seasonId": "current",
            }),
        }

        try:
            resp = self.session.post(
                urljoin(self.BASE_URL, self.API_ENDPOINT),
                data=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])
        except Exception as e:
            print(f"[-] Error fetching match list: {e}")
            return []


def extract_cookies_from_browser():
    """Open browser to get cookies interactively."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[-] Install playwright: pip install playwright && playwright install chromium")
        return None, None

    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║  Login to Wyscout in the browser that opens.                              ║
║  Once logged in and on a match page, close the browser.                   ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    cookies = {}
    access_token = None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        # Capture access token from API calls
        def on_request(request):
            nonlocal access_token
            if "aengine-service" in request.url:
                try:
                    body = request.post_data
                    if body and "access_token" in body:
                        match = re.search(r'"access_token"\s*:\s*"([^"]+)"', body)
                        if match:
                            access_token = match.group(1)
                            print(f"[+] Captured access_token: {access_token[:20]}...")
                except:
                    pass

        page.on("request", on_request)
        page.goto("https://wyscout.hudl.com")

        print("[*] Login and navigate to any match page, then close browser...")

        try:
            page.wait_for_event("close", timeout=0)
        except:
            pass

        # Get cookies
        for cookie in context.cookies():
            cookies[cookie["name"]] = cookie["value"]

        browser.close()

    return cookies, access_token


def interactive_mode():
    """Interactive mode: login, get cookies, fetch URLs."""
    cookies, access_token = extract_cookies_from_browser()

    if not cookies:
        print("[-] No cookies captured")
        return

    print(f"[+] Captured {len(cookies)} cookies")
    if access_token:
        print(f"[+] Access token: {access_token[:20]}...")

    # Save for later use
    with open("wyscout_session.json", "w") as f:
        json.dump({"cookies": cookies, "access_token": access_token}, f, indent=2)
    print("[+] Session saved to wyscout_session.json")

    api = WyscoutAPI(cookies=cookies, access_token=access_token)

    # Get match IDs from user
    print("\nEnter match IDs (comma-separated) or 'q' to quit:")
    while True:
        try:
            user_input = input("> ").strip()
            if user_input.lower() == 'q':
                break

            match_ids = [int(x.strip()) for x in user_input.split(",") if x.strip().isdigit()]

            for match_id in match_ids:
                print(f"[*] Fetching URL for match {match_id}...")
                url = api.get_download_url(match_id)
                if url:
                    print(f"[+] {url}\n")
                time.sleep(1)  # Rate limiting

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[-] Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Wyscout API Downloader")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode with browser login")
    parser.add_argument("--matches", "-m", nargs="+", type=int,
                        help="Match IDs to fetch")
    parser.add_argument("--session", "-s", type=Path, default="wyscout_session.json",
                        help="Session file with cookies/token")
    parser.add_argument("--output", "-o", type=Path, default="download_urls.txt",
                        help="Output file for URLs")
    parser.add_argument("--quality", "-q", default="hd", choices=["hd", "sd"],
                        help="Video quality")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    if not args.matches:
        parser.error("Provide --matches or use --interactive")

    # Load session
    if args.session.exists():
        with open(args.session) as f:
            session_data = json.load(f)
        cookies = session_data.get("cookies", {})
        access_token = session_data.get("access_token")
    else:
        print(f"[-] Session file not found: {args.session}")
        print("[*] Run with --interactive first to login")
        return

    api = WyscoutAPI(cookies=cookies, access_token=access_token)
    urls = []

    for match_id in args.matches:
        print(f"[*] Fetching URL for match {match_id}...")
        url = api.get_download_url(match_id, args.quality)
        if url:
            urls.append(url)
            print(f"[+] Got URL for match {match_id}")
        time.sleep(1)

    # Save URLs
    with open(args.output, "w") as f:
        f.write("\n".join(urls))
    print(f"\n[+] Saved {len(urls)} URLs to {args.output}")
    print(f"[*] Download with: python3 scripts/wyscout_batch.py --urls {args.output} --download")


if __name__ == "__main__":
    main()
