import bcrypt
import logging

logger = logging.getLogger(__name__)

def check_login(username: str, password: str, users_collection) -> bool:
    """Checks if the provided username and password are valid against MongoDB."""
    logger.info(f"Login attempt for user: {username}")
    user = users_collection.find_one({"username": username})
    if user:
        stored_hash = user["password"]

        # Ensure we always have bytes for bcrypt.checkpw
        stored_hash_bytes = stored_hash.encode('utf-8') if isinstance(stored_hash, str) else stored_hash

        try:
            if bcrypt.checkpw(password.encode('utf-8'), stored_hash_bytes):
                logger.info(f"User '{username}' logged in successfully.")
                return True
            else:
                logger.warning(f"Invalid password attempt for user: {username}")
        except Exception as e:
            logger.error(f"Error checking password for user {username}: {e}")
    else:
        logger.warning(f"Login failed, user not found: {username}")
    return False