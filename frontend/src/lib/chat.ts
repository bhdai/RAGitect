/**
 * Chat API client using Vercel AI SDK
 * 
 * Provides streaming chat functionality for RAGitect workspaces.
 * Designed for integration with the Vercel AI SDK's stream handling.
 * 
 * Story 3.0: Streaming Infrastructure - AC2
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Chat message type for conversation history
 */
export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

/**
 * Stream a chat response from the backend SSE endpoint.
 * 
 * Returns the raw Response object for consumption by Vercel AI SDK
 * or manual SSE parsing. The response will be in text/event-stream format
 * with chunks formatted as: data: {"text": "..."}\n\n
 * 
 * @param workspaceId - The workspace to query documents from
 * @param message - The user's message to process
 * @returns Response object with SSE stream
 * 
 * @example
 * ```ts
 * const response = await streamChatResponse(workspaceId, message);
 * const reader = response.body?.getReader();
 * // Process SSE chunks...
 * ```
 */
export async function streamChatResponse(
  workspaceId: string,
  message: string
): Promise<Response> {
  const response = await fetch(
    `${API_URL}/api/v1/workspaces/${workspaceId}/chat/stream`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    }
  );
  return response;
}

/**
 * Parse SSE chunks from a stream response.
 * 
 * This is a utility for manually consuming the SSE stream when not
 * using the Vercel AI SDK's built-in stream handling.
 * 
 * @param response - Response object with SSE stream
 * @yields Parsed text chunks from the stream
 * 
 * @example
 * ```ts
 * const response = await streamChatResponse(workspaceId, message);
 * for await (const chunk of parseSSEStream(response)) {
 *   console.log(chunk); // Process each text chunk
 * }
 * ```
 */
export async function* parseSSEStream(
  response: Response
): AsyncGenerator<string, void, unknown> {
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('Response body is not readable');
  }

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      
      // Process complete SSE events (separated by double newline)
      const events = buffer.split('\n\n');
      buffer = events.pop() || ''; // Keep incomplete event in buffer

      for (const event of events) {
        if (!event.trim()) continue;
        
        // Parse SSE data line
        const dataMatch = event.match(/^data:\s*(.+)$/m);
        if (!dataMatch) continue;

        const data = dataMatch[1].trim();
        
        // Check for stream end marker
        if (data === '[DONE]') {
          return;
        }

        try {
          const parsed = JSON.parse(data);
          if (parsed.text) {
            yield parsed.text;
          }
        } catch {
          // Skip malformed JSON
          console.warn('Failed to parse SSE data:', data);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
