package main

import (
	"context"
	"database/sql"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	_ "github.com/lib/pq"
	"github.com/qonstant/distributed-agent/internal/adapter/postgres"
)

func main() {
	var (
		telegramID = flag.Int64("telegram-id", 0, "optional telegram_id override for the temporary smoke user")
		username   = flag.String("username", "", "optional username override for the temporary smoke user")
	)
	flag.Parse()

	dbURL := strings.TrimSpace(os.Getenv("DB_URL"))
	if dbURL == "" {
		log.Fatal("DB_URL is required")
	}

	smokeTelegramID := *telegramID
	if smokeTelegramID == 0 {
		smokeTelegramID = time.Now().UnixNano()
		if smokeTelegramID < 0 {
			smokeTelegramID = -smokeTelegramID
		}
	}

	smokeUsername := strings.TrimSpace(*username)
	if smokeUsername == "" {
		smokeUsername = fmt.Sprintf("db_smoke_%d", smokeTelegramID)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		log.Fatalf("open postgres connection: %v", err)
	}
	defer db.Close()

	if err := db.PingContext(ctx); err != nil {
		log.Fatalf("ping postgres: %v", err)
	}

	directory, err := postgres.NewAccessDirectory(dbURL)
	if err != nil {
		log.Fatalf("init access directory: %v", err)
	}
	defer directory.Close()

	missingTelegramID := smokeTelegramID + 1
	if err := assertMissing(ctx, directory, missingTelegramID, "before insert"); err != nil {
		log.Fatal(err)
	}

	if err := insertSmokeUser(ctx, db, smokeTelegramID, smokeUsername); err != nil {
		log.Fatalf("insert smoke user: %v", err)
	}
	defer cleanupSmokeUser(context.Background(), db, smokeTelegramID)

	log.Printf("inserted smoke user telegram_id=%d username=@%s", smokeTelegramID, smokeUsername)

	record, found, err := directory.FindByTelegramID(ctx, smokeTelegramID)
	if err != nil {
		log.Fatalf("find inserted smoke user: %v", err)
	}
	if !found {
		log.Fatal("find inserted smoke user: expected record, got not found")
	}
	log.Printf(
		"found smoke user telegram_id=%d allows_now=%t is_blocked=%t expires_at=%v",
		record.TelegramID,
		record.Allows(time.Now().UTC()),
		record.IsBlocked,
		record.AccessExpiresAt,
	)

	if err := deleteSmokeUser(ctx, db, smokeTelegramID); err != nil {
		log.Fatalf("delete smoke user: %v", err)
	}
	log.Printf("deleted smoke user telegram_id=%d", smokeTelegramID)

	if err := assertMissing(ctx, directory, smokeTelegramID, "after delete"); err != nil {
		log.Fatal(err)
	}

	log.Println("db access smoke test passed")
}

func insertSmokeUser(ctx context.Context, db *sql.DB, telegramID int64, username string) error {
	const query = `
		INSERT INTO users (
			telegram_id,
			username,
			first_name,
			last_name,
			is_blocked,
			is_admin,
			access_expires_at
		)
		VALUES ($1, $2, $3, $4, false, false, null)
	`

	_, err := db.ExecContext(ctx, query, telegramID, username, "DB", "Smoke")
	return err
}

func deleteSmokeUser(ctx context.Context, db *sql.DB, telegramID int64) error {
	const query = `DELETE FROM users WHERE telegram_id = $1`

	result, err := db.ExecContext(ctx, query, telegramID)
	if err != nil {
		return err
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if rowsAffected == 0 {
		return errors.New("expected one row to be deleted, got zero")
	}
	return nil
}

func cleanupSmokeUser(ctx context.Context, db *sql.DB, telegramID int64) {
	const query = `DELETE FROM users WHERE telegram_id = $1`
	if _, err := db.ExecContext(ctx, query, telegramID); err != nil {
		log.Printf("cleanup warning for telegram_id=%d: %v", telegramID, err)
	}
}

func assertMissing(ctx context.Context, directory *postgres.AccessDirectory, telegramID int64, label string) error {
	record, found, err := directory.FindByTelegramID(ctx, telegramID)
	if err != nil {
		return fmt.Errorf("check missing user %s: %w", label, err)
	}
	if found {
		return fmt.Errorf("check missing user %s: expected not found, got record %+v", label, record)
	}
	log.Printf("verified missing user %s telegram_id=%d", label, telegramID)
	return nil
}
