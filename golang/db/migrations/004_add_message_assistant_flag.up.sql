ALTER TABLE "messages"
ADD COLUMN "is_assistant" boolean NOT NULL DEFAULT false;

CREATE INDEX "idx_messages_is_assistant" ON "messages" ("is_assistant");

COMMENT ON TABLE "messages" IS 'Stores both user and assistant messages in chronological order.';
COMMENT ON COLUMN "messages"."message_text" IS 'Original message text';
COMMENT ON COLUMN "messages"."is_assistant" IS 'false = user message, true = bot or assistant message';
