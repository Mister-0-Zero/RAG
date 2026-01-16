import sys
import logging
from getpass import getpass

from rag.users import USERS

log = logging.getLogger(__name__)


def authenticate_user(
    user_role_arg: str | None,
    password_arg: str | None,
    default_role: str,
    acl_enabled: bool,
) -> tuple[str, str]:
    """
    Возвращает (username, role).
    Завершает программу при ошибке авторизации.
    """

    if not acl_enabled:
        return "anonymous", "user"

    username = user_role_arg or default_role

    if username not in USERS:
        log.error("Unknown user '%s'", username)
        sys.exit(1)

    expected_password = USERS[username]["password"]
    role = USERS[username]["role"]

    # Если пароль пустой — проверка не нужна (guest)
    if expected_password == "":
        return username, role

    for attempt in range(1, 4):
        password = password_arg or getpass("Enter password: ")

        if password == expected_password:
            return username, role

        log.warning("Invalid password (attempt %d/3)", attempt)
        password_arg = None  # чтобы дальше спрашивало через getpass

    log.error("Authentication failed. Exiting.")
    sys.exit(1)
