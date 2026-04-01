DROP INDEX IF EXISTS "uq_conversations_conversation_key";

ALTER TABLE "conversations"
DROP COLUMN IF EXISTS "conversation_key";
