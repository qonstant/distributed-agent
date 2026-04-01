ALTER TABLE "conversations"
ADD COLUMN IF NOT EXISTS "conversation_key" varchar(255);

UPDATE "conversations"
SET "conversation_key" = 'legacy-' || "id"
WHERE "conversation_key" IS NULL;

ALTER TABLE "conversations"
ALTER COLUMN "conversation_key" SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS "uq_conversations_conversation_key"
ON "conversations" ("conversation_key");

COMMENT ON COLUMN "conversations"."conversation_key" IS 'Stable external conversation key used for async persistence and message grouping.';
