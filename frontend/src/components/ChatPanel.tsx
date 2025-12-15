/**
 * Chat panel for natural language querying of documents
 *
 * Story 3.1: Natural Language Querying
 * - Uses Vercel AI SDK useChat hook for native streaming support
 * - Supports future COT/citation features via message.parts
 *
 * Migrated from custom useChat hook to @ai-sdk/react
 */

'use client';

import { useState, useRef, useEffect, FormEvent, KeyboardEvent } from 'react';
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import { Send, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface ChatPanelProps {
  /** Workspace ID for chat operations */
  workspaceId: string;
}

/**
 * Chat panel component using Vercel AI SDK.
 *
 * Provides:
 * - Native streaming support via AI SDK Data Stream Protocol
 * - Message parts rendering for future COT/citation support
 * - Auto-scroll and loading states
 */
export function ChatPanel({ workspaceId }: ChatPanelProps) {
  const [inputValue, setInputValue] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  const { messages, status, sendMessage, error } = useChat({
    id: `workspace-${workspaceId}`,
    transport: new DefaultChatTransport({
      api: `${API_URL}/api/v1/workspaces/${workspaceId}/chat/stream`,
      // Transform AI SDK message format to backend's ChatRequest format
      prepareSendMessagesRequest: ({ messages }) => {
        // Get the last user message as the current message
        const lastMessage = messages[messages.length - 1];
        const messageText =
          lastMessage?.parts
            ?.filter((p): p is { type: 'text'; text: string } => p.type === 'text')
            .map((p) => p.text)
            .join('') || '';

        // Build chat history from previous messages (excluding the current one)
        const chatHistory = messages.slice(0, -1).map((m) => ({
          role: m.role,
          content:
            m.parts
              ?.filter((p): p is { type: 'text'; text: string } => p.type === 'text')
              .map((p) => p.text)
              .join('') || '',
        }));

        return {
          body: {
            message: messageText,
            chat_history: chatHistory,
          },
        };
      },
    }),
  });

  const isLoading = status === 'streaming' || status === 'submitted';

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const message = inputValue.trim();
    setInputValue('');
    await sendMessage({ text: message });
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as unknown as FormEvent);
    }
  };

  return (
    <div
      data-testid="chat-panel"
      className="flex-1 flex flex-col min-h-0 bg-zinc-50 dark:bg-zinc-900 overflow-hidden"
    >
      {/* Messages area - must have explicit height constraints for scroll */}
      <ScrollArea ref={scrollRef} className="flex-1 min-h-0 p-4">
        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center text-muted-foreground">
            <p>Ask a question about your documents...</p>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={cn(
                  'p-3 rounded-lg max-w-[80%]',
                  message.role === 'user'
                    ? 'ml-auto bg-primary text-primary-foreground'
                    : 'bg-muted'
                )}
              >
                {/* Render message parts - supports future COT/citation via parts */}
                {message.parts.map((part, idx) => {
                  switch (part.type) {
                    case 'text':
                      return (
                        <p key={idx} className="whitespace-pre-wrap">
                          {part.text}
                        </p>
                      );
                    // Future: Handle 'reasoning' parts for chain-of-thought
                    // Future: Handle 'source-document' parts for citations
                    default:
                      return null;
                  }
                })}
              </div>
            ))}
            {isLoading && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Thinking...</span>
              </div>
            )}
          </div>
        )}
      </ScrollArea>

      {/* Error message */}
      {error && (
        <div className="px-4 py-2 bg-destructive/10 text-destructive text-sm">
          {error.message}
        </div>
      )}

      {/* Input area */}
      <form onSubmit={handleSubmit} className="border-t p-4 flex gap-2">
        <Textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your documents..."
          className="min-h-[44px] max-h-32 resize-none"
          disabled={isLoading}
        />
        <Button
          type="submit"
          size="icon"
          disabled={!inputValue.trim() || isLoading}
        >
          {isLoading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </Button>
      </form>
    </div>
  );
}
