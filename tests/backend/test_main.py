from fastapi.testclient import TestClient
import importlib

class DummyCompletions:
    def __init__(self, text_to_return: str):
        self._text_to_return = text_to_return

    def create(self, model, messages, temperature=0.7):
        # Мини-проверки, что эндпоинт прокидывает данные
        assert isinstance(model, str)
        assert model != ""
        assert isinstance(messages, list)
        assert temperature == 0.7

        # Возвращаем объект, похожий на ответ openai-клиента
        message_obj = type("Message", (), {"content": self._text_to_return})()
        choice_obj = type("Choice", (), {"message": message_obj})()
        response_obj = type("Response", (), {"choices": [choice_obj]})()
        return response_obj


class DummyChat:
    def __init__(self, text_to_return: str):
        self.completions = DummyCompletions(text_to_return)


class DummyClient:
    def __init__(self, text_to_return: str):
        self.chat = DummyChat(text_to_return)


def load_main_module(monkeypatch, model_name: str = "mistralai/mistral-7b-instruct"):
    """
    Ваш main.py читает env на импорте (openai_model = os.getenv(...)).
    Поэтому перед импортом подставляем env и импортируем модуль заново.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENAI_MODEL", model_name)

    # Важно: если модуль уже импортирован, перезагрузим его,
    # чтобы подхватил новые env
    import app.main as main_module
    main_module = importlib.reload(main_module)
    return main_module


def test_chat_returns_reply(monkeypatch):
    main_module = load_main_module(monkeypatch, model_name="mistralai/mistral-7b-instruct")

    # Подменяем глобальный client на заглушку
    main_module.client = DummyClient("Тестовый ответ!")

    client = TestClient(main_module.app)
    payload = {"messages": [{"role": "user", "content": "Привет"}], "mode": None}

    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    assert response.json() == {"reply": "Тестовый ответ!"}


def test_chat_uses_messages_from_request(monkeypatch):
    main_module = load_main_module(monkeypatch, model_name="mistralai/mistral-7b-instruct")

    captured = {"messages": None}

    class CapturingCompletions(DummyCompletions):
        def create(self, model, messages, temperature=0.7):
            captured["messages"] = messages
            return super().create(model, messages, temperature)

    class CapturingClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": CapturingCompletions("ok")})()

    main_module.client = CapturingClient()

    client = TestClient(main_module.app)
    payload = {
        "messages": [
            {"role": "system", "content": "Ты помощник"},
            {"role": "user", "content": "Скажи привет"},
        ]
    }

    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    assert response.json()["reply"] == "ok"
    assert captured["messages"] == payload["messages"]


def test_chat_validation_error_without_messages(monkeypatch):
    main_module = load_main_module(monkeypatch)

    client = TestClient(main_module.app)
    response = client.post("/api/chat", json={"mode": None})

    # FastAPI/Pydantic должны вернуть 422
    assert response.status_code == 422


def test_chat_validation_error_messages_not_list(monkeypatch):
    main_module = load_main_module(monkeypatch)

    client = TestClient(main_module.app)
    response = client.post("/api/chat", json={"messages": "not a list"})

    assert response.status_code == 422


def test_index_route_exists(monkeypatch):
    main_module = load_main_module(monkeypatch)

    client = TestClient(main_module.app)
    response = client.get("/")

    # Если файла web/index.html нет в тестовом окружении, может быть 404/500.
    # Поэтому проверяем корректнее: что роут существует в app.routes,
    # а не "файл реально лежит в FS".
    paths = [r.path for r in main_module.app.routes]
    assert "/" in paths