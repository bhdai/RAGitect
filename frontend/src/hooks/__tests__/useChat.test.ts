/**
 * Tests for useChat hook
 *
 * Story 3.1: Natural Language Querying - AC5
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useChat } from '../useChat';

// Mock the chat module
vi.mock('@/lib/chat', () => ({
  streamChatResponse: vi.fn(),
  parseSSEStream: vi.fn(),
}));

import { streamChatResponse, parseSSEStream } from '@/lib/chat';

describe('useChat', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  it('initializes with empty messages', () => {
    const { result } = renderHook(() => useChat({ workspaceId: 'test-123' }));

    expect(result.current.messages).toEqual([]);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it('adds user message when sendMessage is called', async () => {
    const mockResponse = new Response('', { status: 200 });
    vi.mocked(streamChatResponse).mockResolvedValue(mockResponse);

    // Mock parseSSEStream to yield chunks
    vi.mocked(parseSSEStream).mockImplementation(async function* () {
      yield 'Hello!';
    });

    const { result } = renderHook(() => useChat({ workspaceId: 'test-123' }));

    await act(async () => {
      await result.current.sendMessage('Test message');
    });

    expect(result.current.messages[0]).toEqual({
      role: 'user',
      content: 'Test message',
    });
  });

  it('streams assistant response', async () => {
    const mockResponse = new Response('', { status: 200 });
    vi.mocked(streamChatResponse).mockResolvedValue(mockResponse);

    // Mock parseSSEStream to yield chunks
    vi.mocked(parseSSEStream).mockImplementation(async function* () {
      yield 'Hello';
      yield ' World';
    });

    const { result } = renderHook(() => useChat({ workspaceId: 'test-123' }));

    await act(async () => {
      await result.current.sendMessage('Test');
    });

    // Should have user message and assistant response
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[1].role).toBe('assistant');
    expect(result.current.messages[1].content).toBe('Hello World');
  });

  it('sets isLoading during message sending', async () => {
    let resolveResponse: (value: Response) => void;
    const responsePromise = new Promise<Response>((resolve) => {
      resolveResponse = resolve;
    });

    vi.mocked(streamChatResponse).mockReturnValue(responsePromise);
    vi.mocked(parseSSEStream).mockImplementation(async function* () {
      yield 'Done';
    });

    const { result } = renderHook(() => useChat({ workspaceId: 'test-123' }));

    // Start sending
    let sendPromise: Promise<void>;
    act(() => {
      sendPromise = result.current.sendMessage('Hello');
    });

    // Should be loading immediately
    expect(result.current.isLoading).toBe(true);

    // Resolve the response
    await act(async () => {
      resolveResponse!(new Response('', { status: 200 }));
      await sendPromise;
    });

    // Should no longer be loading
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });
  });

  it('handles errors gracefully', async () => {
    const mockResponse = new Response('', { status: 500 });
    vi.mocked(streamChatResponse).mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useChat({ workspaceId: 'test-123' }));

    await act(async () => {
      await result.current.sendMessage('Test');
    });

    expect(result.current.error).not.toBeNull();
    expect(result.current.error).toContain('500');
  });

  it('clears messages when clearMessages is called', async () => {
    const mockResponse = new Response('', { status: 200 });
    vi.mocked(streamChatResponse).mockResolvedValue(mockResponse);
    vi.mocked(parseSSEStream).mockImplementation(async function* () {
      yield 'Response';
    });

    const { result } = renderHook(() => useChat({ workspaceId: 'test-123' }));

    // Add a message
    await act(async () => {
      await result.current.sendMessage('Hello');
    });

    expect(result.current.messages.length).toBeGreaterThan(0);

    // Clear messages
    act(() => {
      result.current.clearMessages();
    });

    expect(result.current.messages).toEqual([]);
    expect(result.current.error).toBeNull();
  });

  it('passes chat history to streamChatResponse', async () => {
    const mockResponse = new Response('', { status: 200 });
    vi.mocked(streamChatResponse).mockResolvedValue(mockResponse);
    vi.mocked(parseSSEStream).mockImplementation(async function* () {
      yield 'First';
    });

    const { result } = renderHook(() => useChat({ workspaceId: 'test-123' }));

    // Send first message
    await act(async () => {
      await result.current.sendMessage('Hello');
    });

    vi.mocked(parseSSEStream).mockImplementation(async function* () {
      yield 'Second';
    });

    // Send second message
    await act(async () => {
      await result.current.sendMessage('How are you?');
    });

    // Check that history was passed
    const calls = vi.mocked(streamChatResponse).mock.calls;
    expect(calls).toHaveLength(2);

    // Second call should include history from first exchange
    const secondCallHistory = calls[1][2]; // Third argument is chat history
    expect(secondCallHistory).toBeDefined();
    expect(secondCallHistory).toHaveLength(2); // user + assistant from first exchange
  });

  it('removes incomplete assistant message on error', async () => {
    const mockResponse = new Response('', { status: 200 });
    vi.mocked(streamChatResponse).mockResolvedValue(mockResponse);

    // Mock parseSSEStream to throw
    vi.mocked(parseSSEStream).mockImplementation(async function* () {
      throw new Error('Stream error');
    });

    const { result } = renderHook(() => useChat({ workspaceId: 'test-123' }));

    await act(async () => {
      await result.current.sendMessage('Test');
    });

    // Should only have user message, assistant message should be removed on error
    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].role).toBe('user');
    expect(result.current.error).not.toBeNull();
  });
});
