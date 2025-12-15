/**
 * Custom hook for managing chat state
 *
 * Story 3.1: Natural Language Querying - AC5
 *
 * Provides stateless chat management with:
 * - Message history in frontend state
 * - Streaming response handling
 * - Error state management
 */

'use client';

import { useState, useCallback } from 'react';
import {
  ChatMessage,
  streamChatResponse,
  parseSSEStream,
} from '@/lib/chat';

export interface UseChatOptions {
  workspaceId: string;
}

export interface UseChatReturn {
  /** Array of chat messages (user and assistant) */
  messages: ChatMessage[];
  /** Whether a message is currently being processed */
  isLoading: boolean;
  /** Error message if last operation failed */
  error: string | null;
  /** Send a new message and stream the response */
  sendMessage: (content: string) => Promise<void>;
  /** Clear all messages and error state */
  clearMessages: () => void;
}

/**
 * Hook for managing chat state with streaming responses
 *
 * @example
 * ```tsx
 * function ChatComponent({ workspaceId }) {
 *   const { messages, isLoading, sendMessage } = useChat({ workspaceId });
 *
 *   return (
 *     <div>
 *       {messages.map((m, i) => <Message key={i} {...m} />)}
 *       <input onSubmit={(text) => sendMessage(text)} disabled={isLoading} />
 *     </div>
 *   );
 * }
 * ```
 */
export function useChat({ workspaceId }: UseChatOptions): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(
    async (content: string) => {
      setError(null);
      setIsLoading(true);

      // Add user message immediately
      const userMessage: ChatMessage = { role: 'user', content };
      setMessages((prev) => [...prev, userMessage]);

      try {
        // Stream assistant response
        const response = await streamChatResponse(
          workspaceId,
          content,
          messages // Pass current history (before adding user message)
        );

        if (!response.ok) {
          throw new Error(`Chat failed: ${response.status}`);
        }

        // Add placeholder for assistant message
        setMessages((prev) => [...prev, { role: 'assistant', content: '' }]);

        // Stream response chunks
        for await (const chunk of parseSSEStream(response)) {
          setMessages((prev) => {
            const updated = [...prev];
            const lastIdx = updated.length - 1;
            updated[lastIdx] = {
              ...updated[lastIdx],
              content: updated[lastIdx].content + chunk,
            };
            return updated;
          });
        }
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : 'Chat failed';
        setError(errorMessage);

        // Remove incomplete assistant message on error (keep user message)
        setMessages((prev) => {
          // If we added an assistant placeholder, remove it
          if (prev.length > 0 && prev[prev.length - 1].role === 'assistant') {
            return prev.slice(0, -1);
          }
          return prev;
        });
      } finally {
        setIsLoading(false);
      }
    },
    [workspaceId, messages]
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return { messages, isLoading, error, sendMessage, clearMessages };
}
