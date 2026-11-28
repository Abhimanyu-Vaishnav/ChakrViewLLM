"""
Mission CV-24H / Release Task: Publish GitHub Release v0.10.0-alpha.1 via GitHub REST API.
Uses Git Credential Manager token to authenticate.
1. Creates release from existing tag v0.10.0-alpha.1 (commit 851f45d).
2. Uploads chakr_38m_conversational_sft.pt as a release asset.
3. Verifies release and asset upload status.
"""

import os
import sys
import json
import urllib.request
import urllib.error
import subprocess
import mimetypes

def get_github_token():
    # Query git credential helper
    p = subprocess.Popen(["git", "credential", "fill"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, _ = p.communicate("protocol=https\nhost=github.com\n")
    for line in out.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

def main():
    token = get_github_token()
    if not token:
        print("ERROR: Could not retrieve GitHub token from git credentials or environment.")
        sys.exit(1)

    owner = "Abhimanyu-Vaishnav"
    repo = "ChakrViewLLM"
    tag_name = "v0.10.0-alpha.1"
    release_name = "ChakrView Conversational Alpha — Experimental Research Preview"
    
    notes_file = "experiments/cv24h_release/release_notes_v0.10.0-alpha.1.md"
    with open(notes_file, "r", encoding="utf-8") as f:
        body_text = f.read()

    print(f"Creating GitHub release for tag {tag_name} on {owner}/{repo}...")
    create_url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    
    payload = {
        "tag_name": tag_name,
        "name": release_name,
        "body": body_text,
        "draft": False,
        "prerelease": True
    }
    
    req = urllib.request.Request(
        create_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "ChakrView-Agent",
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as resp:
            rel_data = json.loads(resp.read().decode())
            print(f"Release successfully created! ID: {rel_data.get('id')}, URL: {rel_data.get('html_url')}")
            upload_url = rel_data.get("upload_url", "").split("{")[0]
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode()
        print(f"HTTPError {e.code}: {e.reason} - {err_msg}")
        # If release already exists, fetch it
        fetch_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag_name}"
        req_fetch = urllib.request.Request(fetch_url, headers={"Authorization": f"Bearer {token}", "User-Agent": "ChakrView-Agent"})
        with urllib.request.urlopen(req_fetch) as resp_f:
            rel_data = json.loads(resp_f.read().decode())
            print(f"Found existing release: ID: {rel_data.get('id')}, URL: {rel_data.get('html_url')}")
            upload_url = rel_data.get("upload_url", "").split("{")[0]

    # Attach checkpoint asset
    ckpt_path = "experiments/cv24h_release/chakr_38m_conversational_sft.pt"
    if os.path.exists(ckpt_path):
        asset_name = "chakr_38m_conversational_sft.pt"
        size_bytes = os.path.getsize(ckpt_path)
        print(f"Uploading asset {asset_name} ({size_bytes / (1024*1024):.2f} MB)...")
        
        # Check if asset already attached
        existing_assets = rel_data.get("assets", [])
        for a in existing_assets:
            if a.get("name") == asset_name:
                print(f"Asset {asset_name} already uploaded: {a.get('browser_download_url')}")
                return

        target_upload_url = f"{upload_url}?name={asset_name}"
        
        with open(ckpt_path, "rb") as f_asset:
            asset_bytes = f_asset.read()

        req_upload = urllib.request.Request(
            target_upload_url,
            data=asset_bytes,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "ChakrView-Agent",
                "Content-Type": "application/octet-stream"
            },
            method="POST"
        )
        with urllib.request.urlopen(req_upload) as resp_up:
            up_data = json.loads(resp_up.read().decode())
            print(f"Asset uploaded successfully! Download URL: {up_data.get('browser_download_url')}")
    else:
        print(f"WARNING: Checkpoint {ckpt_path} not found for attachment.")

if __name__ == "__main__":
    main()
