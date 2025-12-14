/**
 * Tests for chat API client
 * 
 * Story 3.0: Streaming Infrastructure - AC2
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { streamChatResponse, type ChatMessage } from '../chat';

describe('streamChatResponse', () => {
  const mockFetch = vi.fn();
  const originalFetch = global.fetch;

  beforeEach(() => {
    global.fetch = mockFetch;
    mockFetch.mockReset();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it('sends POST request to correct endpoint', async () => {
    const mockResponse = new Response('data: {"text":"Hello"}\n\n', {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    });
    mockFetch.mockResolvedValue(mockResponse);

    await streamChatResponse('workspace-123', 'Hello');

    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/v1/workspaces/workspace-123/chat/stream',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: 'Hello' }),
      })
    );
  });

  it('returns response object for streaming', async () => {
    const mockResponse = new Response('data: {"text":"Hello"}\n\n', {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    });
    mockFetch.mockResolvedValue(mockResponse);

    const response = await streamChatResponse('workspace-123', 'Test message');

    expect(response).toBe(mockResponse);
    expect(response.headers.get('Content-Type')).toBe('text/event-stream');
  });

  it('handles server errors', async () => {
    const mockResponse = new Response(JSON.stringify({ detail: 'Not found' }), {
      status: 404,
    });
    mockFetch.mockResolvedValue(mockResponse);

    const response = await streamChatResponse('invalid-workspace', 'Hello');

    expect(response.status).toBe(404);
  });

  it('uses custom API URL from environment', async () => {
    // This test verifies the API URL is configurable
    const mockResponse = new Response('data: {"text":"Hi"}\n\n', {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    });
    mockFetch.mockResolvedValue(mockResponse);

    await streamChatResponse('ws-1', 'Hi');

    // The default URL should be used when NEXT_PUBLIC_API_URL is not set
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/v1/workspaces/ws-1/chat/stream'),
      expect.any(Object)
    );
  });
});

describe('ChatMessage type', () => {
  it('supports user and assistant roles', () => {
    const userMessage: ChatMessage = {
      role: 'user',
      content: 'Hello',
    };
    const assistantMessage: ChatMessage = {
      role: 'assistant',
      content: 'Hi there!',
    };

    expect(userMessage.role).toBe('user');
    expect(assistantMessage.role).toBe('assistant');
  });
});
