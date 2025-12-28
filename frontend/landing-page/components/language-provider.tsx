"use client"

import type React from "react"
import { createContext, useContext, useState, useEffect } from "react"
import { Button } from "@/components/ui/button"

type Language = "en" | "ru" | "kk"

interface LanguageContextType {
  language: Language
  setLanguage: (lang: Language) => void
  t: (key: string) => string
}

const translations = {
  en: {
    hero_title: "Your Path to",
    hero_title_highlight: "European Education",
    hero_subtitle:
      "We guide international students through every step of university admission in Europe. From application to acceptance, scholarships to visa support — we assist with bureaucracy using AI-powered agents and an AI mentor platform so you don't miss anything.",
    hero_cta: "Start Your Journey",
    universities_title: "Our Students Get Accepted To Top Italian Universities",
    launching_soon: "Launching Soon",
    signup_title: "Join the Waitlist",
    signup_subtitle:
      "Be the first to know when we launch. Get exclusive early access and special benefits for early members.",
    email_placeholder: "your.email@example.com",
    submit_button: "Get Early Access",
    submitting: "Submitting...",
    success_message: "You're on the list! We'll notify you as soon as we launch.",
    error_message: "Something went wrong. Please try again.",
    features_title: "Everything You Need to Get Accepted",
    features_subtitle: "Comprehensive support for your university application journey",
    feature_colleges_title: "Find the Perfect Colleges for You",
    feature_colleges_desc: "Craft a balanced list of target universities based on your grades, goals and preferences.",
    feature_essay_title: "Write a Strong Essay",
    feature_essay_desc: "Generate ideas, drafts and get clear feedback with suggestions how to improve your story.",
    feature_profile_title: "Build a Strong Profile",
    feature_profile_desc:
      "Identify what's missing, find internships and frame your activities in the best way possible.",
    feature_interview_title: "Practice for Every University Interview",
    feature_interview_desc: "Run mock interviews with our agent that knows 8/10 questions you're likely to get.",
    feature_mentor_title: "Be in touch with AI mentor",
    feature_mentor_desc:
      "AI mentor that guides you, reminds deadlines, helps prepare documents, and coordinates submissions.",
    faq_title: "Frequently Asked Questions",
    faq_q1: "What universities do you help with?",
    faq_a1:
      "We specialize in top Italian universities including Politecnico di Milano, Sapienza University of Rome, Università di Bologna, and many more across Europe.",
    faq_q2: "How do you help with scholarships?",
    faq_a2:
      "We identify scholarship opportunities, help prepare compelling applications, and guide you through the entire scholarship application process to maximize your chances of funding.",
    faq_q3: "What about visa and bureaucracy?",
    faq_a3:
      "We assist with student visa applications, residence permits, document authentication, translations, and other bureaucratic procedures — combining expert support with AI agents and an AI mentor platform to keep you on track.",
    faq_q4: "How long does the process take?",
    faq_a4:
      "The timeline varies by university and program, but typically ranges from 3-6 months from application to acceptance. We start working with students 6-12 months before their intended start date.",
    faq_q5: "Do you guarantee acceptance?",
    faq_a5:
      "While we cannot guarantee acceptance, our expert guidance significantly improves your chances. 92% of our students receive offers from their target universities.",
    chat_title: "Questions About Studying Abroad?",
    chat_placeholder: "Ask about admissions, scholarships, visa process...",
    chat_send: "Send",
    footer_tagline: "Empowering students to achieve their European education dreams",
    gdpr_text: "We use cookies to enhance your experience. By continuing, you accept our",
    gdpr_privacy: "Privacy Policy",
    gdpr_accept: "Accept All",
    gdpr_settings: "Cookie Settings",
  },
  ru: {
    hero_title: "Ваш Путь к",
    hero_title_highlight: "Европейскому Образованию",
    hero_subtitle:
      "Мы сопровождаем иностранных студентов на каждом этапе поступления в европейские университеты. От подачи документов до зачисления, от стипендий до визовой поддержки — мы помогаем с бюрократией с помощью ИИ-агентов и платформы-ментора, чтобы вы ничего не пропустили.",
    hero_cta: "Начать Путь",
    universities_title: "Наши Студенты Поступают в Ведущие Итальянские Университеты",
    launching_soon: "Скоро Запуск",
    signup_title: "Присоединяйтесь к Списку Ожидания",
    signup_subtitle:
      "Будьте первыми, кто узнает о нашем запуске. Получите эксклюзивный ранний доступ и специальные преимущества для первых участников.",
    email_placeholder: "ваш.email@пример.com",
    submit_button: "Получить Ранний Доступ",
    submitting: "Отправка...",
    success_message: "Вы в списке! Мы уведомим вас, как только запустимся.",
    error_message: "Что-то пошло не так. Пожалуйста, попробуйте ещё раз.",
    features_title: "Всё Необходимое для Поступления",
    features_subtitle: "Комплексная поддержка на вашем пути к поступлению в университет",
    feature_colleges_title: "Найдите Идеальные Университеты",
    feature_colleges_desc:
      "Составьте сбалансированный список целевых университетов на основе ваших оценок, целей и предпочтений.",
    feature_essay_title: "Напишите Сильное Эссе",
    feature_essay_desc:
      "Генерируйте идеи, черновики и получайте четкую обратную связь с предложениями по улучшению вашей истории.",
    feature_profile_title: "Создайте Сильный Профиль",
    feature_profile_desc:
      "Определите, чего не хватает, найдите стажировки и представьте свою деятельность наилучшим образом.",
    feature_interview_title: "Практикуйте Университетские Интервью",
    feature_interview_desc:
      "Проводите пробные интервью с нашим агентом, который знает 8 из 10 вопросов, которые вам зададут.",
    feature_mentor_title: "Будьте на связи с AI ментором",
    feature_mentor_desc:
      "ИИ-ментор, который направляет вас, напоминает о сроках, помогает подготовить документы и координировать их отправку.",
    faq_title: "Часто Задаваемые Вопросы",
    faq_q1: "С какими университетами вы работаете?",
    faq_a1:
      "Мы специализируемся на ведущих итальянских университетах, включая Politecnico di Milano, Университет Сапиенца в Риме, Болонский университет и многие другие по всей Европе.",
    faq_q2: "Как вы помогаете со стипендиями?",
    faq_a2:
      "Мы находим возможности для получения стипендий, помогаем подготовить убедительные заявки и сопровождаем вас через весь процесс подачи заявок, чтобы максимально увеличить ваши шансы на финансирование.",
    faq_q3: "А как насчёт визы и бюрократии?",
    faq_a3:
      "Мы помогаем при подаче заявлений на студенческую визу, разрешения на проживание, легализации документов, переводов и во всех бюрократических процедурах — сочетая экспертную поддержку с ИИ-агентами и платформой-ментором, чтобы вы были в курсе.",
    faq_q4: "Сколько времени занимает процесс?",
    faq_a4:
      "Сроки варьируются в зависимости от университета и программы, но обычно составляют от 3 до 6 месяцев от подачи до зачисления. Мы начинаем работать со студентами за 6-12 месяцев до предполагаемого начала обучения.",
    faq_q5: "Вы гарантируете поступление?",
    faq_a5:
      "Хотя мы не можем гарантировать поступление, наша экспертная поддержка значительно повышает ваши шансы. 92% наших студентов получают предложения от выбранных университетов.",
    chat_title: "Вопросы об Учёбе за Рубежом?",
    chat_placeholder: "Спросите о поступлении, стипендиях, визовом процессе...",
    chat_send: "Отправить",
    footer_tagline: "Помогаем студентам осуществить их мечты об европейском образовании",
    gdpr_text: "Мы используем файлы cookie для улучшения вашего опыта. Продолжая, вы принимаете нашу",
    gdpr_privacy: "Политику конфиденциальности",
    gdpr_accept: "Принять всё",
    gdpr_settings: "Настройки cookie",
  },
  kk: {
    hero_title: "Сіздің Жолыңыз",
    hero_title_highlight: "Еуропалық Білімге",
    hero_subtitle:
      "Біз халықаралық студенттерді Еуропадағы университеттерге түсу процесінің әрбір қадамын басшылыққа аламыз. Өтінімнен қабылдауға дейін, стипендиядан виза қолдауға дейін — біз бюрократияға ИИ-негізделген агенттер мен менторлық платформа арқылы көмек көрсетеміз, сондықтан сіз ештеңені жіберіп алмайсыз.",
    hero_cta: "Жолды Бастау",
    universities_title: "Біздің Студенттер Италияның Үздік Университеттеріне Түседі",
    launching_soon: "Жақында Іске Қосылады",
    signup_title: "Күту Тізіміне Қосылыңыз",
    signup_subtitle:
      "Біз іске қосылған кезде біріншілерден болып біліңіз. Эксклюзивті ерте қол жеткізу және алғашқы мүшелерге арналған арнайы артықшылықтар алыңыз.",
    email_placeholder: "сіздің.email@мысал.com",
    submit_button: "Ерте Қол Жеткізу",
    submitting: "Жіберілуде...",
    success_message: "Сіз тізімдесіз! Біз іске қосылған кезде сізге хабарлаймыз.",
    error_message: "Бірдеңе дұрыс болмады. Қайталап көріңіз.",
    features_title: "Қабылдану үшін Барлық Қажеттілік",
    features_subtitle: "Университетке өтінім беру жолындағы кешенді қолдау",
    feature_colleges_title: "Өзіңізге Ыңғайлы Колледждерді Табыңыз",
    feature_colleges_desc:
      "Бағаларыңыз, мақсаттарыңыз және қалауларыңыз негізінде мақсатты университеттердің теңдестірілген тізімін жасаңыз.",
    feature_essay_title: "Күшті Эссе Жазыңыз",
    feature_essay_desc:
      "Идеяларды, жобаларды жасаңыз және тарихыңызды жақсарту бойынша ұсыныстармен нақты кері байланыс алыңыз.",
    feature_profile_title: "Күшті Профиль Құрыңыз",
    feature_profile_desc:
      "Не жетіспейтінін анықтаңыз, тәжірибелерді табыңыз және белсенділікті мүмкіндігінше жақсы түрде көрсетіңіз.",
    feature_interview_title: "Әрбір Университет Сұхбатына Дайындалыңыз",
    feature_interview_desc:
      "Сізге қойылуы мүмкін 10 сұрақтың 8-ін білетін агентіміздің көмегімен сынақ сұхбаттарын өткізіңіз.",
    feature_mentor_title: "AI менторымен байланыста болыңыз",
    feature_mentor_desc:
      "Сізге бағыт-бағдар беретін, мерзімдерді еске түсіретін, құжаттарды дайындауға және жіберуге көмектесетін ИИ менторы.",
    faq_title: "Жиі Қойылатын Сұрақтар",
    faq_q1: "Қандай университеттерге көмектесесіз?",
    faq_a1:
      "Біз Politecnico di Milano, Рим Сапиенца университеті, Болон университеті және Еуропадағы көптеген басқа үздік итальян университеттеріне маманданамыз.",
    faq_q2: "Стипендияларға қалай көмектесесіз?",
    faq_a2:
      "Біз стипендия мүмкіндіктерін анықтаймыз, сендіретін өтінімдерді дайындауға көмектесеміз және қаржыландыру мүмкіндіктерін арттыру үшін стипендияға өтінім беру процесінің барлық кезеңінде жетекшілік етеміз.",
    faq_q3: "Виза және бюрократия туралы не айтасыз?",
    faq_a3:
      "Біз студенттік виза өтінімдері, тұруға рұқсаттар, құжаттарды куәландыру, аудармалар және Еуропада оқу үшін қажет барлық бюрократиялық рәсімдер бойынша көмек көрсетеміз — сараптамалық қолдау мен ИИ-агенттер мен менторлық платформа арқылы сіз ештеңені жіберіп алмайсыз.",
    faq_q4: "Процесс қанша уақыт алады?",
    faq_a4:
      "Мерзім университет пен бағдарламаға байланысты өзгереді, бірақ әдетте өтінімнен қабылдауға дейін 3-6 ай аралығында болады. Біз студенттермен олардың оқуды бастауға арналған мерзімінен 6-12 ай бұрын жұмыс жасауды бастаймыз.",
    faq_q5: "Қабылдануға кепілдік бересіз бе?",
    faq_a5:
      "Біз қабылдануға кепілдік бере алмасақ та, біздің сарапшы көмегіміз мүмкіндіктеріңізді айтарлықтай арттырады. Біздің студенттердің 92%-ы таңдаған университеттерінен ұсыныстар алады.",
    chat_title: "Шетелде Оқу Туралы Сұрақтар?",
    chat_placeholder: "Қабылдау, стипендиялар, виза процесі туралы сұраңыз...",
    chat_send: "Жіберу",
    footer_tagline: "Студенттердің еуропалық білім арманын орындауға көмектесу",
    gdpr_text: "Біз тәжірибеңізді жақсарту үшін cookie файлдарын пайдаланамыз. Жалғастыра отырып, сіз біздің",
    gdpr_privacy: "Құпиялылық саясатын",
    gdpr_accept: "Барлығын қабылдау",
    gdpr_settings: "Cookie параметрлері",
  },
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined)

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [language, setLanguage] = useState<Language>("en")

  useEffect(() => {
    const saved = localStorage.getItem("nomadmit_language") as Language
    if (saved && (saved === "en" || saved === "ru" || saved === "kk")) {
      setLanguage(saved)
    }
  }, [])

  const handleSetLanguage = (lang: Language) => {
    setLanguage(lang)
    localStorage.setItem("nomadmit_language", lang)
  }

  const t = (key: string) => {
    return translations[language][key as keyof typeof translations.en] || key
  }

  return (
    <LanguageContext.Provider value={{ language, setLanguage: handleSetLanguage, t }}>
      <div className="fixed top-4 right-4 z-50 flex gap-2">
        <Button
          variant={language === "en" ? "default" : "outline"}
          size="sm"
          onClick={() => handleSetLanguage("en")}
          className="font-medium"
        >
          EN
        </Button>
        <Button
          variant={language === "ru" ? "default" : "outline"}
          size="sm"
          onClick={() => handleSetLanguage("ru")}
          className="font-medium"
        >
          RU
        </Button>
        <Button
          variant={language === "kk" ? "default" : "outline"}
          size="sm"
          onClick={() => handleSetLanguage("kk")}
          className="font-medium"
        >
          KK
        </Button>
      </div>
      {children}
    </LanguageContext.Provider>
  )
}

export function useLanguage() {
  const context = useContext(LanguageContext)
  if (!context) throw new Error("useLanguage must be used within LanguageProvider")
  return context
}
