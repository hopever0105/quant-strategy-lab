"""KRX 요청 제한 장치 (모든 퀀트·지수·가치평가 코드에서 반드시 먼저 import)

사용법 (스크립트 맨 위, pykrx·FinanceDataReader import 전에):
    import krx_guard          # quant 폴더가 sys.path에 있을 때
    krx_guard.install()

동작:
- krx.co.kr 로 가는 모든 HTTP 요청(pykrx, FinanceDataReader의 KRX 조회 포함)을 가로채서
  1) 요청 사이 최소 간격(MIN_INTERVAL초)을 강제로 둔다
  2) 하루 최대 요청 수(DAILY_LIMIT)를 넘으면 중지한다
  3) 연속 실패(빈 응답·오류)가 MAX_FAIL번이면 그날은 KRX 요청을 잠근다(다음 날 자동 해제)
- 사용량은 quant/logs/krx_usage.json 에 날짜별로 남는다.
- 잠긴 상태에서 요청하면 KRXBlocked 예외가 나고 요청은 보내지 않는다.

숫자는 아래 상수에서만 바꾼다. 사용자 승인 없이 올리지 않는다.
"""
import json
import os
import threading
import time
from datetime import date
from pathlib import Path

import requests

MIN_INTERVAL = 2.0     # KRX 요청 사이 최소 간격(초)
DAILY_LIMIT = 500      # 하루 최대 KRX 요청 수
MAX_FAIL = 3           # 연속 실패 허용 횟수 → 넘으면 그날 잠금

USAGE_FILE = Path(os.environ.get('QUANT_HOME', Path(__file__).resolve().parent)) / 'logs' / 'krx_usage.json'
_lock = threading.Lock()
_last = 0.0
_installed = False


class KRXBlocked(RuntimeError):
    pass


def _load():
    try:
        return json.loads(USAGE_FILE.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _save(d):
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    USAGE_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')


def _today():
    d = _load()
    k = date.today().isoformat()
    t = d.setdefault(k, {'count': 0, 'fail_streak': 0, 'locked': False, 'note': ''})
    return d, t


def status():
    """오늘 KRX 사용량을 돌려준다."""
    return _today()[1]


def _is_krx(url):
    return 'krx.co.kr' in str(url)


def _check_body(resp):
    """빈 응답이나 차단 페이지면 실패로 본다."""
    if resp.status_code != 200:
        return False
    body = resp.content or b''
    if len(body) < 50:
        return False
    low = body[:2000].lower()
    if b'block' in low or b'\xec\xb0\xa8\xeb\x8b\xa8' in low:   # '차단'
        return False
    return True


def _wrap(orig):
    def guarded(self, method, url, *args, **kwargs):
        if not _is_krx(url):
            return orig(self, method, url, *args, **kwargs)
        global _last
        with _lock:
            d, t = _today()
            if t['locked']:
                raise KRXBlocked(f"오늘 KRX 요청이 잠겨 있음: {t['note']}")
            if t['count'] >= DAILY_LIMIT:
                t['locked'] = True
                t['note'] = f'하루 한도 {DAILY_LIMIT}건 도달'
                _save(d)
                raise KRXBlocked(t['note'])
            wait = MIN_INTERVAL - (time.time() - _last)
            if wait > 0:
                time.sleep(wait)
            _last = time.time()
            t['count'] += 1
            _save(d)
        try:
            resp = orig(self, method, url, *args, **kwargs)
            ok = _check_body(resp)
        except Exception:
            resp, ok = None, False
        with _lock:
            d, t = _today()
            t['fail_streak'] = 0 if ok else t['fail_streak'] + 1
            if t['fail_streak'] >= MAX_FAIL:
                t['locked'] = True
                t['note'] = f'연속 실패 {MAX_FAIL}회 - 차단 의심, 오늘 KRX 요청 중지'
            _save(d)
            if t['locked'] and not ok:
                raise KRXBlocked(t['note'])
        if resp is None:
            raise KRXBlocked('KRX 요청 오류')
        return resp
    return guarded


def install():
    """requests 전체에 KRX 가드를 건다. 여러 번 불러도 한 번만 적용."""
    global _installed
    if _installed:
        return
    requests.Session.request = _wrap(requests.Session.request)
    _installed = True


if __name__ == '__main__':
    print(json.dumps(status(), ensure_ascii=False))
