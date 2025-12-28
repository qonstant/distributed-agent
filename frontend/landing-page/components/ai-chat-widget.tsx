"use client"

import { useState, useEffect, useRef } from "react"
import { useLanguage } from "@/components/language-provider"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import { MessageSquare, X, Send } from "lucide-react"

interface Message {
  role: "user" | "assistant"
  content: string
  timestamp: number
}

export function AIChatWidget() {
  const { t } = useLanguage()
  const [isOpen, setIsOpen] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const saved = localStorage.getItem("nomadmit_chat")
    if (saved) {
      try {
        setMessages(JSON.parse(saved))
      } catch (e) {
        console.error("[v0] Failed to load chat history", e)
      }
    }
  }, [])

  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem("nomadmit_chat", JSON.stringify(messages))
    }
  }, [messages])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const mockResponses = {
    en: [
      "nomadmit connects you with verified local hosts and digital nomads worldwide. We create authentic experiences and meaningful connections!",
      "We're launching in major digital nomad hubs like Bali, Lisbon, Medellín, and more. Join the waitlist to get notified when we launch in your area!",
      "All hosts go through a strict verification process to ensure safety and quality. We verify identity, background, and community reviews.",
      "You can connect with local hosts for coworking, events, cultural experiences, and more. It's all about building genuine connections!",
    ],
    ru: [
      "nomadmit соединяет вас с проверенными местными хозяевами и цифровыми кочевниками по всему миру. Мы создаём аутентичный опыт и значимые связи!",
      "Мы запускаемся в основных центрах цифровых кочевников, таких как Бали, Лиссабон, Медельин и других. Присоединяйтесь к списку ожидания, чтобы получить уведомление о запуске в вашем регионе!",
      "Все хозяева проходят строгий процесс верификации для обеспечения безопасности и качества. Мы проверяем личность, историю и отзывы сообщества.",
      "Вы можете общаться с местными хозяевами для совместной работы, мероприятий, культурного опыта и многого другого. Всё дело в создании настоящих связей!",
    ],
  }

  const handleSend = () => {
    if (!input.trim()) return

    const userMessage: Message = {
      role: "user",
      content: input,
      timestamp: Date.now(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsTyping(true)

    setTimeout(
      () => {
        const { language } = JSON.parse(localStorage.getItem("nomadmit_language") || '"en"')
        const responses = mockResponses[language as "en" | "ru"] || mockResponses.en
        const randomResponse = responses[Math.floor(Math.random() * responses.length)]

        const assistantMessage: Message = {
          role: "assistant",
          content: randomResponse,
          timestamp: Date.now(),
        }

        setMessages((prev) => [...prev, assistantMessage])
        setIsTyping(false)
      },
      1000 + Math.random() * 1000,
    )
  }

  return (
    <>
      {/* Floating Button */}
      <Button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 h-14 w-14 rounded-full shadow-lg"
        size="icon"
      >
        <MessageSquare className="h-6 w-6" />
      </Button>

      {/* Chat Widget */}
      {isOpen && (
        <Card className="fixed bottom-24 right-6 w-96 h-[500px] shadow-2xl flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b">
            <h3 className="font-semibold">{t("chat_title")}</h3>
            <Button variant="ghost" size="icon" onClick={() => setIsOpen(false)}>
              <X className="h-4 w-4" />
            </Button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[80%] rounded-lg px-4 py-2 ${
                    msg.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"
                  }`}
                >
                  {msg.content}
                </div>
              </div>
            ))}
            {isTyping && (
              <div className="flex justify-start">
                <div className="bg-muted text-muted-foreground rounded-lg px-4 py-2">
                  <div className="flex gap-1">
                    <span className="animate-bounce">•</span>
                    <span className="animate-bounce delay-100">•</span>
                    <span className="animate-bounce delay-200">•</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-4 border-t flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              placeholder={t("chat_placeholder")}
            />
            <Button onClick={handleSend} size="icon">
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </Card>
      )}
    </>
  )
}
