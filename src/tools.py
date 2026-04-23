import re


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    message = f"Lead captured successfully: {name}, {email}, {platform}"
    print(message)
    return message


def is_valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))
