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
    Returns (username, role).
    Exits the program on authorization failure.
    """

    if not acl_enabled:
        return "anonymous", "user"

    username = user_role_arg or default_role

    if username not in USERS:
        log.error("Unknown user '%s'", username)
        sys.exit(1)

    expected_password = USERS[username]["password"]
    role = USERS[username]["role"]

    # If the password is empty, no check is needed (e.g., for a guest user)
    if expected_password == "":
        return username, role

    for attempt in range(1, 4):
        password = password_arg or getpass("Enter password: ")

        if password == expected_password:
            return username, role

        log.warning("Invalid password (attempt %d/3)", attempt)
        # so that it prompts for password again using getpass
        password_arg = None

    log.error("Authentication failed. Exiting.")
    sys.exit(1)
