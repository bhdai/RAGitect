/**
 * Chat types and utilities for Vercel AI SDK integration
 *
 * Note: The useChat hook from @ai-sdk/react handles SSE parsing via
 * DefaultChatTransport. This file provides shared types only.
 */

export const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Chat message type for conversation history
 * @deprecated Use UIMessage from @ai-sdk/react instead for AI SDK components
 */
export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}
