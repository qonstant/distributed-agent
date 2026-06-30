DROP INDEX IF EXISTS "idx_messages_is_assistant";

ALTER TABLE "messages"
DROP COLUMN IF EXISTS "is_assistant";

COMMENT ON TABLE "messages" IS 'Stores only user messages for the MVP.';
COMMENT ON COLUMN "messages"."message_text" IS 'Original user message text';
