/**
 * Chat panel for natural language querying of documents
 *
 * Natural Language Querying
 * - Uses Vercel AI SDK useChat hook for native streaming support
 * - Supports future COT/citation features via message.parts
 *
 * Migrated from custom useChat hook to @ai-sdk/react
 */

'use client';

import { useState, useRef, useEffect, useMemo, FormEvent, KeyboardEvent } from 'react';
// Note: useRef is still used for selectedProviderRef
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import { ArrowUp, Plus, Clock, MessageSquare } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from '@/components/ai-elements/conversation';
import { Shimmer } from '@/components/ai-elements/shimmer';

import { ChatProviderSelector } from '@/components/ChatProviderSelector';
import { AssistantMessage } from '@/components/AssistantMessage';
import { useProviderSelectionContext } from '@/contexts/ProviderSelectionContext';
import { cn } from '@/lib/utils';
import { buildCitationMap, type CitationData } from '@/types/citation';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface ChatPanelProps {
  /** Workspace ID for chat operations */
  workspaceId: string;
  /** Callback when user clicks a citation badge*/
  onCitationClick?: (citation: CitationData) => void;
}

/**
 * Chat panel component using Vercel AI SDK.
 *
 * Provides:
 * - Native streaming support via AI SDK Data Stream Protocol
 * - Message parts rendering for future COT/citation support
 * - Auto-scroll and loading states
 */
export function ChatPanel({ workspaceId, onCitationClick }: ChatPanelProps) {
  const [inputValue, setInputValue] = useState('');
  const { selectedProvider } = useProviderSelectionContext();

  // Use a ref to always have the current selectedProvider value in callbacks
  const selectedProviderRef = useRef(selectedProvider);
  useEffect(() => {
    selectedProviderRef.current = selectedProvider;
  }, [selectedProvider]);

  // Create transport with a stable reference. The prepareSendMessagesRequest callback
  // accesses selectedProviderRef.current only when invoked (during sendMessage event),
  // not during the render phase. This pattern prevents transport recreation when
  // selectedProvider changes, which would reset chat state.
  /* eslint-disable react-hooks/refs -- ref accessed in callback, not during render */
  const transport = useMemo(() => {
    // This function is called during message send, not during render
    const prepareSendMessagesRequest = ({ messages }: { messages: Array<{ role: string; parts?: Array<{ type: string; text?: string }> }> }) => {
      // Safe to access ref here - this runs during event handling, not render
      const currentProvider = selectedProviderRef.current;

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
          // Use ref to get current provider value (avoids stale closure)
          provider: currentProvider,
        },
      };
    };

    return new DefaultChatTransport({
      api: `${API_URL}/api/v1/workspaces/${workspaceId}/chat/stream`,
      // Transform AI SDK message format to backend's ChatRequest format
      prepareSendMessagesRequest,
    });
  }, [workspaceId]);
  /* eslint-enable react-hooks/refs */

  const { messages, status, sendMessage, error } = useChat({
    id: `workspace-${workspaceId}`,
    transport,
  });

  const isLoading = status === 'streaming' || status === 'submitted';
  // Only show thinking indicator when submitted but not yet streaming
  const showThinkingIndicator = status === 'submitted';

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
      {/* Messages area - Conversation handles auto-scroll with StickToBottom */}
      <Conversation className="flex-1 min-h-0">
        <ConversationContent className="gap-6 p-0 min-h-full">
          {messages.length === 0 ? (
            <ConversationEmptyState
              icon={<MessageSquare className="size-12" />}
              title="Ask a question about your documents"
              description="Start a conversation to explore your uploaded documents."
            />
          ) : (
            <div className="w-full max-w-3xl mx-auto px-4 pt-6 pb-24 space-y-6">
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
                    <AssistantMessage
                      message={message}
                      onCitationClick={onCitationClick}
                    />
                  )}
                </div>
              ))}

              {/* Animated thinking indicator - only show before streaming starts */}
              {showThinkingIndicator && (
                <div className="flex flex-col gap-1 items-start w-full" data-testid="thinking-indicator">
                  <span className="text-xs font-medium text-muted-foreground px-1">
                    Assistant
                  </span>
                  <Shimmer className="text-sm" duration={1.5}>
                    Searching your documents...
                  </Shimmer>
                </div>
              )}
            </div>
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      {/* Error message */}
      {error && (
        <div className="px-4 py-2 bg-destructive/10 text-destructive text-sm">
          {error.message}
        </div>
      )}

      {/* Input area - floats into message area for visual effect */}
      <div className="flex-shrink-0 px-4 pb-4 -mt-16 relative z-10">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
          <div className="rounded-2xl border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-800 shadow-lg transition-all overflow-hidden">
            {/* Textarea - full width at top, NOT disabled during streaming so user can type ahead */}
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
