-- Guest feedback is intentionally stored separately from registered-user chats.
-- It contains no account ID and is linked only to an opaque browser-session token.
CREATE TABLE IF NOT EXISTS guest_support_chat (
    guest_chat_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    guest_token VARCHAR(128) NOT NULL,
    guest_name VARCHAR(100) NULL,
    message TEXT NOT NULL,
    sender ENUM('guest', 'admin') NOT NULL,
    sent_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (guest_chat_id),
    INDEX idx_guest_support_conversation (guest_token, sent_at, guest_chat_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
