/**
 * Tests for ChatPanel component with Vercel AI SDK
 *
 * Story 3.0: Streaming Infrastructure - AC3
 * Story 3.1: Natural Language Querying - AC1, AC4
 *
 * These tests mock the @ai-sdk/react useChat hook
 */

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ChatPanel } from '../ChatPanel';

// Mock the @ai-sdk/react useChat hook
vi.mock('@ai-sdk/react', () => ({
  useChat: vi.fn(),
}));

// Mock the ai package for DefaultChatTransport
vi.mock('ai', () => ({
  DefaultChatTransport: class MockDefaultChatTransport {
    constructor() { }
  },
}));

import { useChat } from '@ai-sdk/react';

/** Helper to create mock useChat return value */
function createMockUseChatReturn(overrides: {
  messages?: unknown[];
  status?: string;
  error?: Error | undefined;
  sendMessage?: ReturnType<typeof vi.fn>;
}) {
  return {
    messages: overrides.messages ?? [],
    status: overrides.status ?? 'ready',
    error: overrides.error,
    sendMessage: overrides.sendMessage ?? vi.fn(),
    setMessages: vi.fn(),
    reload: vi.fn(),
    regenerate: vi.fn(),
    stop: vi.fn(),
    resumeStream: vi.fn(),
    addToolResult: vi.fn(),
    addToolOutput: vi.fn(),
    clearError: vi.fn(),
    input: '',
    setInput: vi.fn(),
    handleSubmit: vi.fn(),
    handleInputChange: vi.fn(),
    isLoading: overrides.status === 'streaming' || overrides.status === 'submitted',
    data: undefined,
    metadata: undefined,
    id: 'test',
  } as unknown as ReturnType<typeof useChat>;
}

describe('ChatPanel', () => {
  const mockSendMessage = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useChat).mockReturnValue(createMockUseChatReturn({
      sendMessage: mockSendMessage,
    }));
  });

  it('renders the chat panel with input', () => {
    render(<ChatPanel workspaceId="test-workspace" />);

    expect(screen.getByTestId('chat-panel')).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText(/ask a question/i)
    ).toBeInTheDocument();
  });

  it('shows empty state when no messages', () => {
    render(<ChatPanel workspaceId="test-workspace" />);

    expect(
      screen.getByText(/ask a question about your documents/i)
    ).toBeInTheDocument();
  });

  it('renders user messages', () => {
    vi.mocked(useChat).mockReturnValue(createMockUseChatReturn({
      messages: [
        {
          id: 'msg-1',
          role: 'user',
          parts: [{ type: 'text', text: 'Hello there!' }],
        },
      ],
      sendMessage: mockSendMessage,
    }));

    render(<ChatPanel workspaceId="test-workspace" />);

    expect(screen.getByText('Hello there!')).toBeInTheDocument();
  });

  it('renders assistant messages with parts', () => {
    vi.mocked(useChat).mockReturnValue(createMockUseChatReturn({
      messages: [
        {
          id: 'msg-1',
          role: 'user',
          parts: [{ type: 'text', text: 'What is Python?' }],
        },
        {
          id: 'msg-2',
          role: 'assistant',
          parts: [{ type: 'text', text: 'Python is a programming language.' }],
        },
      ],
      sendMessage: mockSendMessage,
    }));

    render(<ChatPanel workspaceId="test-workspace" />);

    expect(screen.getByText('What is Python?')).toBeInTheDocument();
    expect(
      screen.getByText('Python is a programming language.')
    ).toBeInTheDocument();
  });

  it('shows loading indicator when streaming', () => {
    vi.mocked(useChat).mockReturnValue(createMockUseChatReturn({
      messages: [
        {
          id: 'msg-1',
          role: 'user',
          parts: [{ type: 'text', text: 'Hello' }],
        },
      ],
      status: 'streaming',
      sendMessage: mockSendMessage,
    }));

    render(<ChatPanel workspaceId="test-workspace" />);

    expect(screen.getByText(/thinking/i)).toBeInTheDocument();
  });

  it('shows error message when error occurs', () => {
    vi.mocked(useChat).mockReturnValue(createMockUseChatReturn({
      status: 'error',
      error: new Error('Connection failed'),
      sendMessage: mockSendMessage,
    }));

    render(<ChatPanel workspaceId="test-workspace" />);

    expect(screen.getByText('Connection failed')).toBeInTheDocument();
  });

  it('sends message when form is submitted', async () => {
    const user = userEvent.setup();
    mockSendMessage.mockResolvedValue(undefined);

    render(<ChatPanel workspaceId="test-workspace" />);

    const input = screen.getByPlaceholderText(/ask a question/i);
    await user.type(input, 'What is RAG?');

    const submitButton = screen.getByRole('button');
    await user.click(submitButton);

    expect(mockSendMessage).toHaveBeenCalledWith({ text: 'What is RAG?' });
  });

  it('sends message when Enter key is pressed', async () => {
    const user = userEvent.setup();
    mockSendMessage.mockResolvedValue(undefined);

    render(<ChatPanel workspaceId="test-workspace" />);

    const input = screen.getByPlaceholderText(/ask a question/i);
    await user.type(input, 'Hello{Enter}');

    expect(mockSendMessage).toHaveBeenCalledWith({ text: 'Hello' });
  });

  it('does not send empty messages', async () => {
    const user = userEvent.setup();

    render(<ChatPanel workspaceId="test-workspace" />);

    const submitButton = screen.getByRole('button');
    await user.click(submitButton);

    expect(mockSendMessage).not.toHaveBeenCalled();
  });

  it('disables input when streaming', () => {
    vi.mocked(useChat).mockReturnValue(createMockUseChatReturn({
      status: 'streaming',
      sendMessage: mockSendMessage,
    }));

    render(<ChatPanel workspaceId="test-workspace" />);

    const input = screen.getByPlaceholderText(/ask a question/i);
    expect(input).toBeDisabled();
  });

  it('has proper flex styling', () => {
    render(<ChatPanel workspaceId="test-workspace" />);

    const panel = screen.getByTestId('chat-panel');
    expect(panel).toHaveClass('flex-1');
    expect(panel).toHaveClass('flex');
    expect(panel).toHaveClass('flex-col');
  });

  it('has proper overflow handling', () => {
    render(<ChatPanel workspaceId="test-workspace" />);

    const panel = screen.getByTestId('chat-panel');
    expect(panel).toHaveClass('overflow-hidden');
  });
});
