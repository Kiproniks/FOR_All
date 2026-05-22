import glob
import json
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
INBOX = ROOT / 'backend' / 'media' / 'books' / 'inbox'
STATE_PATH = ROOT / 'run_logs' / 'fb2_uploaded_state.json'
LOG_PATH = ROOT / 'run_logs' / 'fb2_uploader.log'
BASE = 'http://127.0.0.1:8000'
EMAIL = 'student@local.local'
PASSWORD = 'Student123!'

INBOX.mkdir(parents=True, exist_ok=True)
STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(f'[{ts}] {message}\n')


def load_state() -> set[str]:
    if not STATE_PATH.exists():
        return set()
    try:
        data = json.loads(STATE_PATH.read_text(encoding='utf-8'))
        if isinstance(data, list):
            return set(str(x) for x in data)
    except Exception:
        pass
    return set()


def save_state(uploaded: set[str]) -> None:
    STATE_PATH.write_text(json.dumps(sorted(uploaded), ensure_ascii=False, indent=2), encoding='utf-8')


def get_token(session: requests.Session) -> str | None:
    reg = session.post(
        f'{BASE}/api/auth/register/',
        json={'email': EMAIL, 'password': PASSWORD, 'password_repeat': PASSWORD},
        timeout=30,
    )
    if reg.status_code in (200, 201):
        return reg.json().get('token')

    login = session.post(
        f'{BASE}/api/auth/login/',
        json={'email': EMAIL, 'password': PASSWORD},
        timeout=30,
    )
    if login.status_code == 200:
        return login.json().get('token')
    log(f'Auth failed: register={reg.status_code} login={login.status_code}')
    return None


def main() -> None:
    uploaded = load_state()
    session = requests.Session()
    token = None

    while True:
        if token is None:
            try:
                token = get_token(session)
            except Exception as exc:
                log(f'Auth error: {exc}')
                token = None

        book_files = sorted(glob.glob(str(INBOX / '*.fb2'))) + sorted(glob.glob(str(INBOX / '*.pdf')))
        for file_path in book_files:
            if file_path in uploaded:
                continue
            if token is None:
                break
            try:
                with open(file_path, 'rb') as f:
                    ext = Path(file_path).suffix.lower()
                    content_type = 'application/pdf' if ext == '.pdf' else 'application/xml'
                    resp = session.post(
                        f'{BASE}/api/books/upload/',
                        headers={'Authorization': f'Token {token}'},
                        files=[('files', (os.path.basename(file_path), f, content_type))],
                        timeout=180,
                    )
                if resp.status_code in (200, 201):
                    uploaded.add(file_path)
                    save_state(uploaded)
                    log(f'Uploaded {os.path.basename(file_path)} status={resp.status_code}')
                elif resp.status_code == 401:
                    token = None
                    log(f'Token expired while uploading {os.path.basename(file_path)}')
                elif 400 <= resp.status_code < 500:
                    # Client-side validation errors (unsupported/bad file) should not be retried forever.
                    uploaded.add(file_path)
                    save_state(uploaded)
                    log(
                        f'Skipped {os.path.basename(file_path)} due to client error '
                        f'status={resp.status_code} body={resp.text[:500]}'
                    )
                else:
                    log(f'Upload failed {os.path.basename(file_path)} status={resp.status_code} body={resp.text[:500]}')
            except Exception as exc:
                log(f'Upload exception {os.path.basename(file_path)}: {exc}')
        time.sleep(3)


if __name__ == '__main__':
    main()
