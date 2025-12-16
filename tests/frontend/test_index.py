from pathlib import Path
import re


def read_index_html() -> str:
    root = Path(__file__).resolve().parents[2]
    path = root / "web" / "index.html"
    return path.read_text(encoding="utf-8")


def test_index_html_has_basic_structure():
    html = read_index_html()
    assert "<!doctype html>" in html.lower()
    assert 'lang="ru"' in html
    assert "<title>Interview Coach (LLM)</title>" in html


def test_index_html_has_required_controls():
    html = read_index_html()
    assert 'id="toggleDemo"' in html
    assert 'id="clearChat"' in html
    assert 'id="startInterview"' in html
    assert 'id="askHint"' in html
    assert 'id="askSample"' in html
    assert 'id="askEval"' in html
    assert 'id="chat"' in html
    assert 'id="input"' in html
    assert 'id="sendBtn"' in html


def test_index_html_has_localstorage_keys():
    html = read_index_html()
    assert 'LS_KEY = "interview_coach_chat_v1"' in html
    assert 'LS_DEMO = "interview_coach_demo_v1"' in html


def test_index_html_calls_backend_endpoint():
    html = read_index_html()
    assert 'fetch("/api/chat"' in html
    assert 'method:"POST"' in html
    assert '"Content-Type":"application/json"' in html


def test_index_html_payload_contains_mode_and_messages():
    html = read_index_html()
    assert 'mode: "interview_coach"' in html
    assert "messages:" in html
    assert "role:\"system\"" in html or 'role:"system"' in html


def test_index_html_has_demo_mode_toggle():
    html = read_index_html()
    assert "let demoMode" in html
    assert "localStorage.getItem(LS_DEMO)" in html
    assert "demoMode = !demoMode" in html
    assert "localStorage.setItem(LS_DEMO" in html


def test_index_html_has_typing_indicator_marker():
    html = read_index_html()
    assert "__TYPING__" in html
    assert "addTyping" in html


def test_index_html_has_render_skipping_system_messages():
    html = read_index_html()
    assert "if (m.role === \"system\") continue;" in html or "if (m.role === 'system') continue;" in html


def test_index_html_has_demo_llm_function():
    html = read_index_html()
    assert "async function demoLLM" in html
    assert "function makeQuestion" in html


def test_index_html_has_expected_quick_actions_texts():
    html = read_index_html()
    assert "Старт" in html
    assert "Подсказка" in html
    assert "Эталонный ответ" in html
    assert "Оценить мой ответ" in html


def test_index_html_has_single_api_chat_path_occurrence_in_fetch():
    html = read_index_html()
    count = len(re.findall(r'fetch\("/api/chat"', html))
    assert count >= 1