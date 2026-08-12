def support_insert_message(execute_db, user_id, message, sender):
    if sender not in {"user", "admin"}:
        raise ValueError("Invalid support message sender.")
    execute_db(
        "INSERT INTO support_chat (user_id, message, sender) VALUES (%s, %s, %s)",
        (user_id, message, sender),
    )


def support_get_messages_for_user(query_db, user_id):
    return query_db(
        """
        SELECT chat_id, user_id, message, sender, sent_at
        FROM support_chat
        WHERE user_id = %s
        ORDER BY sent_at ASC, chat_id ASC
        """,
        (user_id,),
    )


def support_get_admin_messages(query_db, user_id):
    return query_db(
        "SELECT sc.chat_id, sc.user_id, sc.message, sc.sender, sc.sent_at,"
        " u.name AS user_name"
        " FROM support_chat sc"
        " JOIN user u ON sc.user_id = u.user_id"
        " WHERE sc.user_id = %s"
        " ORDER BY sc.sent_at ASC, sc.chat_id ASC",
        (user_id,),
    )


def support_get_admin_chats(query_db):
    return query_db(
        """
        SELECT sc.chat_id, sc.user_id, sc.message, sc.sender, sc.sent_at,
               u.name AS user_name, u.email
        FROM support_chat sc
        JOIN user u ON sc.user_id = u.user_id
        ORDER BY sc.sent_at ASC, sc.chat_id ASC
        """,
    )


def support_get_chat_users(query_db):
    return query_db(
        """
        SELECT DISTINCT u.user_id, u.name, u.email
        FROM support_chat sc
        JOIN user u ON sc.user_id = u.user_id
        ORDER BY u.name ASC, u.user_id ASC
        """,
    )


GUEST_SUPPORT_SCHEMA = """
CREATE TABLE IF NOT EXISTS guest_support_chat (
    guest_chat_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    guest_token VARCHAR(128) NOT NULL,
    guest_name VARCHAR(100) NULL,
    message TEXT NOT NULL,
    sender ENUM('guest', 'admin') NOT NULL,
    sent_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (guest_chat_id),
    INDEX idx_guest_support_conversation (guest_token, sent_at, guest_chat_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def support_ensure_guest_schema(execute_db):
    execute_db(GUEST_SUPPORT_SCHEMA)


def support_insert_guest_message(execute_db, guest_token, guest_name, message, sender):
    if sender not in {"guest", "admin"}:
        raise ValueError("Invalid guest support message sender.")
    execute_db(
        """
        INSERT INTO guest_support_chat (guest_token, guest_name, message, sender)
        VALUES (%s, %s, %s, %s)
        """,
        (guest_token, guest_name or None, message, sender),
    )


def support_get_guest_messages(query_db, guest_token):
    return query_db(
        """
        SELECT guest_chat_id, guest_token, guest_name, message, sender, sent_at
        FROM guest_support_chat
        WHERE guest_token = %s
        ORDER BY sent_at ASC, guest_chat_id ASC
        """,
        (guest_token,),
    )


def support_get_guest_conversations(query_db):
    return query_db(
        """
        SELECT guest_token,
               COALESCE(MAX(NULLIF(guest_name, '')), 'Guest') AS guest_name,
               MAX(sent_at) AS last_message_at,
               COUNT(*) AS message_count
        FROM guest_support_chat
        GROUP BY guest_token
        ORDER BY last_message_at DESC, guest_token ASC
        """
    )
