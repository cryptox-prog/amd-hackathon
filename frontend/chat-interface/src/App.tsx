import { useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type Message = {
  id: number
  author: 'assistant' | 'user'
  text: string
  timestamp: string
}

type ApiResponse = {
  response: string
}

const API_URL =
  import.meta.env.VITE_API_URL

const suggestions = [
  'What is Australia’s capital?',
  'Solve a quick inventory problem',
  'Classify a mixed review',
  'Summarize renewable energy blockers',
  'Extract names and places',
  'Fix a buggy Python function',
  'Solve a pet logic puzzle',
  'Write a second-largest function',
]

function getTimestamp() {
  return new Intl.DateTimeFormat('en', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(new Date())
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [draft, setDraft] = useState('')
  const [isThinking, setIsThinking] = useState(false)

  const canSend = draft.trim().length > 0 && !isThinking
  const hasMessages = messages.length > 0

  async function queueAssistantReply(userText: string) {
    setIsThinking(true)

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userText }),
      })

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      const data = (await response.json()) as ApiResponse

      setMessages((current) => [
        ...current,
        {
          id: Date.now(),
          author: 'assistant',
          text: data.response,
          timestamp: getTimestamp(),
        },
      ])
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: Date.now(),
          author: 'assistant',
          text:
            error instanceof Error
              ? `I could not reach the backend: ${error.message}`
              : 'I could not reach the backend.',
          timestamp: getTimestamp(),
        },
      ])
    } finally {
      setIsThinking(false)
    }
  }

  function sendMessage(text = draft) {
    const trimmedText = text.trim()
    if (!trimmedText || isThinking) return

    setMessages((current) => [
      ...current,
      {
        id: Date.now(),
        author: 'user',
        text: trimmedText,
        timestamp: getTimestamp(),
      },
    ])
    setDraft('')
    queueAssistantReply(trimmedText)
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    sendMessage()
  }

  return (
    <main className="chat-shell">
      <section className="chat-panel" aria-label="Chat conversation">
        <div className="message-list" aria-live="polite">
          {!hasMessages && !isThinking && (
            <div className="welcome-state">
              <div>
                <p className="eyebrow">Workspace assistant</p>
                <h1>How can I help?</h1>
              </div>

              <div className="suggestions" aria-label="Suggested prompts">
                {suggestions.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    onClick={() => sendMessage(suggestion)}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message) => (
            <article
              key={message.id}
              className={`message message-${message.author}`}
            >
              <div className="message-avatar" aria-hidden="true">
                {message.author === 'assistant' ? 'AI' : 'You'}
              </div>
              <div className="message-content">
                <div className="message-meta">
                  <strong>
                    {message.author === 'assistant' ? 'Assistant' : 'You'}
                  </strong>
                  <time>{message.timestamp}</time>
                </div>
                <p>{message.text}</p>
              </div>
            </article>
          ))}

          {isThinking && (
            <article className="message message-assistant">
              <div className="message-avatar" aria-hidden="true">
                AI
              </div>
              <div className="message-content typing">
                <span />
                <span />
                <span />
              </div>
            </article>
          )}
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <label htmlFor="message-input">Message</label>
          <div className="composer-row">
            <textarea
              id="message-input"
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Ask about tasks, results, or next steps..."
              rows={2}
            />
            <button type="submit" disabled={!canSend}>
              Send
            </button>
          </div>
        </form>
      </section>
    </main>
  )
}

export default App
