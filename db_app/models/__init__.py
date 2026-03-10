# Import all models so they are registered on Base.metadata before create_all() is called.
from db_app.models.chat_history import ChatHistory  # noqa: F401
from db_app.models.logs import AppLog  # noqa: F401
from db_app.models.user import User  # noqa: F401
from db_app.models.user_settings import UserSettings  # noqa: F401
