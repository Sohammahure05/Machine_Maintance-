import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta

import storage
from config import LOCK_ATTEMPTS, LOCK_MINUTES, MIN_PASSWORD

ITERATIONS = 200_000
USERNAME = re.compile(r"^[A-Za-z0-9_]{3,32}$")
STAMP = "%Y-%m-%d %H:%M:%S"


def hash_password(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    return salt, hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)


def register(username, password, company):
    username, company = username.strip(), company.strip()
    if not USERNAME.match(username):
        return False, "Username must be 3 to 32 letters, digits or underscores."
    if not company:
        return False, "Enter a company name."
    if len(password) < MIN_PASSWORD or password.lower() == username.lower():
        return False, f"Password must be at least {MIN_PASSWORD} characters and different from the username."
    if not (any(c.isalpha() for c in password) and any(c.isdigit() for c in password)):
        return False, "Password must contain at least one letter and one digit."
    salt, digest = hash_password(password)
    if not storage.create_user(username, company, salt, digest):
        return False, "That username is already taken."
    return True, "Account created. You can log in now."


def authenticate(username, password):
    user = storage.get_user(username.strip())
    if user is None:
        hash_password(password)
        return None, "Invalid username or password."
    if user["locked_until"] and datetime.strptime(user["locked_until"], STAMP) > datetime.now():
        return None, f"Too many failed attempts. Try again after {user['locked_until']}."
    _, digest = hash_password(password, user["salt"])
    if hmac.compare_digest(digest, user["pw_hash"]):
        storage.set_attempts(user["id"], 0, None)
        return {"id": user["id"], "username": user["username"], "company": user["company"]}, ""
    failed = user["failed"] + 1
    locked = None
    if failed >= LOCK_ATTEMPTS:
        locked = (datetime.now() + timedelta(minutes=LOCK_MINUTES)).strftime(STAMP)
        failed = 0
    storage.set_attempts(user["id"], failed, locked)
    return None, "Invalid username or password."
