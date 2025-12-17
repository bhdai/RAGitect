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
import { ArrowUp, Plus, Clock } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { MessageResponse } from '@/components/ai-elements/message';
import { ChatProviderSelector } from '@/components/ChatProviderSelector';
import { useProviderSelection } from '@/hooks/useProviderSelection';
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
  const { selectedProvider } = useProviderSelection();

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
            // Include provider selection for per-request provider override
            provider: selectedProvider,
          },
        };
      },
    }),
  });

  const isLoading = status === 'streaming' || status === 'submitted';
  // Only show thinking indicator when submitted but not yet streaming
  const showThinkingIndicator = status === 'submitted';

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
      className="h-full flex flex-col min-h-0 overflow-hidden"
    >
      {/* Messages area - scrollable with flex-1 to take remaining space */}
      <ScrollArea ref={scrollRef} className="flex-1 min-h-0">
        {messages.length === 0 ? (
          // Empty state - placeholder for future summary/suggestion prompts
          <div className="h-full flex flex-col items-center justify-center text-muted-foreground p-8">
            <div className="max-w-md text-center space-y-4">
              <p className="text-lg">Ask a question about your documents</p>
              <p className="text-sm opacity-70">
                Start a conversation to explore your uploaded documents.
              </p>
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto px-4 pt-6 pb-20 space-y-6">
            {messages.map((message) => (
              <div
                key={message.id}
                className={cn(
                  'flex flex-col gap-1',
                  message.role === 'user' ? 'items-end' : 'items-start'
                )}
              >
                {/* Role label */}
                <span className="text-xs font-medium text-muted-foreground px-1">
                  {message.role === 'user' ? 'You' : 'Assistant'}
                </span>

                {/* Message content - bubble for user, full width for assistant */}
                {message.role === 'user' ? (
                  <div className="max-w-[85%] px-4 py-3 rounded-2xl bg-primary text-primary-foreground rounded-br-md">
                    {message.parts.map((part, idx) => {
                      if (part.type === 'text') {
                        return (
                          <p key={idx} className="whitespace-pre-wrap text-sm">
                            {part.text}
                          </p>
                        );
                      }
                      return null;
                    })}
                  </div>
                ) : (
                  <div className="w-full">
                    {message.parts.map((part, idx) => {
                      if (part.type === 'text') {
                        // Using AI Elements MessageResponse for proper markdown rendering
                        // with syntax highlighting and copy-to-clipboard on code blocks
                        return (
                          <MessageResponse key={idx}>
                            {part.text}
                          </MessageResponse>
                        );
                      }
                      // Future: Handle 'reasoning' parts for chain-of-thought
                      // Future: Handle 'source-document' parts for citations
                      return null;
                    })}
                  </div>
                )}
              </div>
            ))}

            {/* Animated thinking indicator - only show before streaming starts */}
            {showThinkingIndicator && (
              <div className="flex flex-col gap-1 items-start" data-testid="thinking-indicator">
                <span className="text-xs font-medium text-muted-foreground px-1">
                  Assistant
                </span>
                <div className="flex gap-1.5 items-center h-5 px-1">
                  <span className="h-2 w-2 rounded-full bg-zinc-400 dark:bg-zinc-500 animate-bounce" />
                  <span className="h-2 w-2 rounded-full bg-zinc-400 dark:bg-zinc-500 animate-bounce [animation-delay:150ms]" />
                  <span className="h-2 w-2 rounded-full bg-zinc-400 dark:bg-zinc-500 animate-bounce [animation-delay:300ms]" />
                </div>
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

      {/* Input area - anchored at bottom, floating effect with negative margin */}
      <div className="flex-shrink-0 px-4 pb-4 -mt-6 relative z-10">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
          <div className="rounded-2xl border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-800 shadow-lg transition-all overflow-hidden">
            {/* Textarea - full width at top */}
            <Textarea
              value={inputValue}
              onChange={(e) => {
                setInputValue(e.target.value);
                // Auto-resize textarea
                const target = e.target as HTMLTextAreaElement;
                target.style.height = 'auto';
                target.style.height = `${Math.min(target.scrollHeight, 288)}px`; // max ~12 lines (24px * 12)
              }}
              onKeyDown={handleKeyDown}
              placeholder="How can I help you today?"
              className="min-h-[44px] max-h-72 resize-none border-0 bg-transparent dark:bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 py-3 px-4 rounded-t-2xl text-sm leading-6 w-full shadow-none"
              disabled={isLoading}
              rows={1}
            />

            {/* Bottom toolbar row */}
            <div className="flex items-center justify-between px-2 pt-1 pb-2">
              {/* Left side - Plus and Clock buttons */}
              <div className="flex items-center gap-1">
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 rounded-lg text-muted-foreground hover:text-foreground hover:bg-zinc-100 dark:hover:bg-zinc-700"
                  disabled={isLoading}
                >
                  <Plus className="h-4 w-4" />
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 rounded-lg text-muted-foreground hover:text-foreground hover:bg-zinc-100 dark:hover:bg-zinc-700"
                  disabled={isLoading}
                >
                  <Clock className="h-4 w-4" />
                </Button>
              </div>

              {/* Right side - Model selector and Submit button */}
              <div className="flex items-center gap-2">
                <ChatProviderSelector />
                <Button
                  type="submit"
                  size="icon"
                  className="h-8 w-8 rounded-lg bg-primary hover:bg-primary/90 disabled:bg-zinc-300 dark:disabled:bg-zinc-600 disabled:opacity-100 transition-colors"
                  disabled={!inputValue.trim() || isLoading}
                >
                  <ArrowUp className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
