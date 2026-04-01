package telegram

import (
	"context"
	"fmt"
	"html"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/go-telegram/bot"
	"github.com/go-telegram/bot/models"
)

var thinkingFramesLatin = []string{
	"Thinking.",
	"Thinking..",
	"Thinking...",
}

var thinkingFramesCyrillic = []string{
	"Думаю.",
	"Думаю..",
	"Думаю...",
}

type progressFact struct {
	Text   string
	Source string
}

var europeFactsLatin = []progressFact{
	{
		Text:   "The national anthem of Spain has no words.",
		Source: "Spain",
	},
	{
		Text:   "There is a Victor Hugo Street in every town in France.",
		Source: "France",
	},
	{
		Text:   "Rome, also known as the Eternal City, is almost 3,000 years old, and it has been Italy’s capital since 1871.",
		Source: "Italy",
	},
	{
		Text:   "Hamburgers got their name from Hamburg, Germany’s second-largest city.",
		Source: "Germany",
	},
	{
		Text:   "In the UK, if you reach your 100th birthday or your 70th wedding anniversary, you get a personalized card from the King.",
		Source: "United Kingdom",
	},
	{
		Text:   "The official Twitter account @sweden is given to a random citizen every week to manage.",
		Source: "Sweden",
	},
	{
		Text:   "Greek has been spoken for more than 3,000 years, making it one of the oldest languages in Europe.",
		Source: "Greece",
	},
	{
		Text:   "In Switzerland, there are more banks than dentists.",
		Source: "Switzerland",
	},
	{
		Text:   "Iceland has no mosquitoes.",
		Source: "Iceland",
	},
	{
		Text:   "The Eiffel Tower can grow a few centimeters taller in summer because metal expands in heat.",
		Source: "France",
	},
	{
		Text:   "Venice has more than 400 bridges.",
		Source: "Italy",
	},
	{
		Text:   "Liechtenstein is one of the few countries with no airport.",
		Source: "Liechtenstein",
	},
	{
		Text:   "Finland has around 3 million saunas.",
		Source: "Finland",
	},
	{
		Text:   "The Colosseum in Rome could hold around 50,000 spectators.",
		Source: "Italy",
	},
	{
		Text:   "Denmark is often ranked among the happiest countries in the world.",
		Source: "Denmark",
	},
	{
		Text:   "Norway once knighted a penguin called Sir Nils Olav.",
		Source: "Norway",
	},
	{
		Text:   "The Czech Republic is famous for having one of the highest beer consumption rates per person in the world.",
		Source: "Czech Republic",
	},
	{
		Text:   "Portugal is home to one of the oldest bookstores in the world, in Lisbon.",
		Source: "Portugal",
	},
	{
		Text:   "The Netherlands has more bicycles than people.",
		Source: "Netherlands",
	},
	{
		Text:   "France is the most visited country in the world.",
		Source: "France",
	},
	{
		Text:   "Malta has three UNESCO World Heritage Sites despite being one of Europe’s smallest countries.",
		Source: "Malta",
	},
	{
		Text:   "Bulgaria is one of the oldest countries in Europe and has kept the same name for over 1,300 years.",
		Source: "Bulgaria",
	},
	{
		Text:   "Croatia has a sea organ in Zadar that makes music using waves.",
		Source: "Croatia",
	},
	{
		Text:   "Estonia has one of the highest numbers of startups per person in Europe.",
		Source: "Estonia",
	},
	{
		Text:   "In Scotland, the national animal is the unicorn.",
		Source: "Scotland",
	},
	{
		Text:   "Italy has more UNESCO World Heritage Sites than any other country.",
		Source: "Italy",
	},
	{
		Text:   "Belgium is famous for having more than 1,000 varieties of beer.",
		Source: "Belgium",
	},
	{
		Text:   "Germany has over 20,000 castles.",
		Source: "Germany",
	},
	{
		Text:   "Austria is home to the world’s oldest still-operating zoo.",
		Source: "Austria",
	},
	{
		Text:   "Slovenia has a large underground canyon system inside the Škocjan Caves.",
		Source: "Slovenia",
	},
}

var europeFactsCyrillic = []progressFact{
	{
		Text:   "У национального гимна Испании нет слов.",
		Source: "Испания",
	},
	{
		Text:   "Почти в каждом городе Франции есть улица Виктора Гюго.",
		Source: "Франция",
	},
	{
		Text:   "Риму, который называют Вечным городом, почти 3 000 лет, а столицей Италии он стал в 1871 году.",
		Source: "Италия",
	},
	{
		Text:   "Гамбургеры получили свое название в честь Гамбурга, второго по величине города Германии.",
		Source: "Германия",
	},
	{
		Text:   "В Великобритании на 100-летие или 70-летие свадьбы можно получить персональную открытку от короля.",
		Source: "Великобритания",
	},
	{
		Text:   "Официальный аккаунт @sweden каждую неделю передают случайному гражданину для ведения.",
		Source: "Швеция",
	},
	{
		Text:   "На греческом языке говорят уже более 3 000 лет, что делает его одним из древнейших языков Европы.",
		Source: "Греция",
	},
	{
		Text:   "В Швейцарии банков больше, чем стоматологов.",
		Source: "Швейцария",
	},
	{
		Text:   "В Исландии нет комаров.",
		Source: "Исландия",
	},
	{
		Text:   "Летом Эйфелева башня может становиться на несколько сантиметров выше, потому что металл расширяется от жары.",
		Source: "Франция",
	},
	{
		Text:   "В Венеции более 400 мостов.",
		Source: "Италия",
	},
	{
		Text:   "Лихтенштейн — одна из немногих стран, у которых нет аэропорта.",
		Source: "Лихтенштейн",
	},
	{
		Text:   "В Финляндии около 3 миллионов саун.",
		Source: "Финляндия",
	},
	{
		Text:   "Колизей в Риме мог вмещать около 50 000 зрителей.",
		Source: "Италия",
	},
	{
		Text:   "Данию часто включают в число самых счастливых стран мира.",
		Source: "Дания",
	},
	{
		Text:   "В Норвегии однажды посвятили в рыцари пингвина по имени Сэр Нильс Олав.",
		Source: "Норвегия",
	},
	{
		Text:   "Чехия известна одним из самых высоких показателей потребления пива на душу населения в мире.",
		Source: "Чехия",
	},
	{
		Text:   "В Лиссабоне, в Португалии, находится один из старейших книжных магазинов в мире.",
		Source: "Португалия",
	},
	{
		Text:   "В Нидерландах велосипедов больше, чем людей.",
		Source: "Нидерланды",
	},
	{
		Text:   "Франция — самая посещаемая страна в мире.",
		Source: "Франция",
	},
	{
		Text:   "На Мальте три объекта Всемирного наследия ЮНЕСКО, хотя это одна из самых маленьких стран Европы.",
		Source: "Мальта",
	},
	{
		Text:   "Болгария — одна из древнейших стран Европы и сохраняет одно и то же название уже более 1 300 лет.",
		Source: "Болгария",
	},
	{
		Text:   "В хорватском Задаре есть морской орган, который создает музыку с помощью волн.",
		Source: "Хорватия",
	},
	{
		Text:   "В Эстонии один из самых высоких показателей стартапов на душу населения в Европе.",
		Source: "Эстония",
	},
	{
		Text:   "В Шотландии национальным животным считается единорог.",
		Source: "Шотландия",
	},
	{
		Text:   "В Италии больше объектов Всемирного наследия ЮНЕСКО, чем в любой другой стране.",
		Source: "Италия",
	},
	{
		Text:   "Бельгия славится более чем 1 000 сортов пива.",
		Source: "Бельгия",
	},
	{
		Text:   "В Германии более 20 000 замков.",
		Source: "Германия",
	},
	{
		Text:   "В Австрии находится старейший в мире зоопарк, который до сих пор работает.",
		Source: "Австрия",
	},
	{
		Text:   "В Словении в пещерах Шкоцьян находится огромная подземная каньонная система.",
		Source: "Словения",
	},
}

const thinkingUpdateInterval = 1200 * time.Millisecond

type ProgressMessage struct {
	bot       *bot.Bot
	chatID    int64
	messageID int
	cancel    context.CancelFunc
	done      chan struct{}
	stopOnce  sync.Once
	frames    []string
	fact      progressFact
}

func StartProgressMessage(ctx context.Context, b *bot.Bot, chatID int64, text string) *ProgressMessage {
	if b == nil {
		return nil
	}
	frames := thinkingFramesForText(text)
	facts := progressFactsForText(text)
	fact := pickProgressFact(facts, time.Now().UnixNano())

	msg, err := b.SendMessage(ctx, &bot.SendMessageParams{
		ChatID:    chatID,
		Text:      renderProgressText(frames, fact, 0),
		ParseMode: models.ParseModeHTML,
	})
	if err != nil {
		return nil
	}

	runCtx, cancel := context.WithCancel(ctx)
	progress := &ProgressMessage{
		bot:       b,
		chatID:    chatID,
		messageID: msg.ID,
		cancel:    cancel,
		done:      make(chan struct{}),
		frames:    frames,
		fact:      fact,
	}

	go progress.animate(runCtx)
	return progress
}

func (p *ProgressMessage) animate(ctx context.Context) {
	defer close(p.done)

	ticker := time.NewTicker(thinkingUpdateInterval)
	defer ticker.Stop()

	frameIdx := 1
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			text := renderProgressText(p.frames, p.fact, frameIdx)
			frameIdx++
			_, _ = p.bot.EditMessageText(ctx, &bot.EditMessageTextParams{
				ChatID:    p.chatID,
				MessageID: p.messageID,
				Text:      text,
				ParseMode: models.ParseModeHTML,
			})
		}
	}
}

func (p *ProgressMessage) stop() {
	if p == nil {
		return
	}

	p.stopOnce.Do(func() {
		if p.cancel != nil {
			p.cancel()
		}
		if p.done != nil {
			<-p.done
		}
	})
}

func (p *ProgressMessage) Replace(ctx context.Context, text string) error {
	if p == nil {
		return nil
	}

	p.stop()
	text = sanitizeUTF8(strings.TrimSpace(text))
	if text == "" {
		return p.Delete(ctx)
	}

	_, err := p.bot.EditMessageText(ctx, &bot.EditMessageTextParams{
		ChatID:    p.chatID,
		MessageID: p.messageID,
		Text:      text,
	})
	return err
}

func (p *ProgressMessage) Delete(ctx context.Context) error {
	if p == nil {
		return nil
	}

	p.stop()
	_, err := p.bot.DeleteMessage(ctx, &bot.DeleteMessageParams{
		ChatID:    p.chatID,
		MessageID: p.messageID,
	})
	return err
}

func thinkingFramesForText(text string) []string {
	hasLatin := false
	for _, r := range text {
		if !unicode.IsLetter(r) {
			continue
		}
		if unicode.In(r, unicode.Cyrillic) {
			return thinkingFramesCyrillic
		}
		if unicode.In(r, unicode.Latin) {
			hasLatin = true
		}
	}
	if hasLatin {
		return thinkingFramesLatin
	}
	return thinkingFramesLatin
}

func progressFactsForText(text string) []progressFact {
	hasLatin := false
	for _, r := range text {
		if !unicode.IsLetter(r) {
			continue
		}
		if unicode.In(r, unicode.Cyrillic) {
			return europeFactsCyrillic
		}
		if unicode.In(r, unicode.Latin) {
			hasLatin = true
		}
	}
	if hasLatin {
		return europeFactsLatin
	}
	return europeFactsLatin
}

func pickProgressFact(facts []progressFact, seed int64) progressFact {
	if len(facts) == 0 {
		return progressFact{}
	}
	if seed < 0 {
		seed = -seed
	}
	return facts[int(seed%int64(len(facts)))]
}

func renderProgressText(frames []string, fact progressFact, step int) string {
	header := thinkingFramesLatin[0]
	if len(frames) > 0 {
		header = frames[step%len(frames)]
	}
	if strings.TrimSpace(fact.Text) == "" {
		return html.EscapeString(header)
	}

	introLabel := "Interesting fact:"
	sourceLabel := "Source"
	if len(frames) > 0 && frames[0] == thinkingFramesCyrillic[0] {
		introLabel = "Интересный факт:"
		sourceLabel = "Источник"
	}
	return fmt.Sprintf(
		"%s\n\n<b>%s</b>\n<blockquote>%s</blockquote>\n<i>%s: %s</i>",
		html.EscapeString(header),
		html.EscapeString(introLabel),
		html.EscapeString(fact.Text),
		html.EscapeString(sourceLabel),
		html.EscapeString(fact.Source),
	)
}
