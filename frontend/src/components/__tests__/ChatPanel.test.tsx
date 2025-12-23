/**
 * Tests for ChatPanel component with Vercel AI SDK
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

// Mock the useProviderSelectionContext hook
vi.mock('@/contexts/ProviderSelectionContext', () => ({
  useProviderSelectionContext: vi.fn(() => ({
    selectedProvider: 'openai',
    selectProvider: vi.fn(),
    providers: [{ providerName: 'openai', displayName: 'OpenAI', model: 'gpt-4o', isActive: true }],
    currentProvider: { providerName: 'openai', displayName: 'OpenAI', model: 'gpt-4o', isActive: true },
    isLoading: false,
    error: null,
  })),
}));

// Mock the ChatProviderSelector component
vi.mock('@/components/ChatProviderSelector', () => ({
  ChatProviderSelector: () => <div data-testid="chat-provider-selector">gpt-4o</div>,
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
      screen.getByPlaceholderText(/how can i help you today/i)
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

  it('shows loading indicator when submitted', () => {
    vi.mocked(useChat).mockReturnValue(createMockUseChatReturn({
      messages: [
        {
          id: 'msg-1',
          role: 'user',
          parts: [{ type: 'text', text: 'Hello' }],
        },
      ],
      status: 'submitted',
      sendMessage: mockSendMessage,
    }));

    render(<ChatPanel workspaceId="test-workspace" />);

    expect(screen.getByTestId('thinking-indicator')).toBeInTheDocument();
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

    const input = screen.getByPlaceholderText(/how can i help you today/i);
    await user.type(input, 'What is RAG?');

    const submitButton = screen.getAllByRole('button').find(btn => btn.getAttribute('type') === 'submit')!;
    await user.click(submitButton);

    expect(mockSendMessage).toHaveBeenCalledWith({ text: 'What is RAG?' });
  });

  it('sends message when Enter key is pressed', async () => {
    const user = userEvent.setup();
    mockSendMessage.mockResolvedValue(undefined);

    render(<ChatPanel workspaceId="test-workspace" />);

    const input = screen.getByPlaceholderText(/how can i help you today/i);
    await user.type(input, 'Hello{Enter}');

    expect(mockSendMessage).toHaveBeenCalledWith({ text: 'Hello' });
  });

  it('does not send empty messages', async () => {
    const user = userEvent.setup();

    render(<ChatPanel workspaceId="test-workspace" />);

    // The submit button should be disabled when input is empty
    const submitButton = screen.getAllByRole('button').find(btn => btn.getAttribute('type') === 'submit')!;
    await user.click(submitButton);

    expect(mockSendMessage).not.toHaveBeenCalled();
  });

  it('allows typing but disables send button when streaming', () => {
    vi.mocked(useChat).mockReturnValue(createMockUseChatReturn({
      status: 'streaming',
      sendMessage: mockSendMessage,
    }));

    render(<ChatPanel workspaceId="test-workspace" />);

    // User can still type while streaming (type-ahead)
    const input = screen.getByPlaceholderText(/how can i help you today/i);
    expect(input).not.toBeDisabled();

    // But cannot send until streaming completes - find by type="submit"
    const form = input.closest('form')!;
    const submitButton = form.querySelector('button[type="submit"]');
    expect(submitButton).toBeDisabled();
  });

  it('has proper flex styling', () => {
    render(<ChatPanel workspaceId="test-workspace" />);

    const panel = screen.getByTestId('chat-panel');
    expect(panel).toHaveClass('h-full');
    expect(panel).toHaveClass('flex');
    expect(panel).toHaveClass('flex-col');
  });

  it('has proper overflow handling', () => {
    render(<ChatPanel workspaceId="test-workspace" />);

    const panel = screen.getByTestId('chat-panel');
    expect(panel).toHaveClass('overflow-hidden');
  });

  describe('Chat Management (AC1, AC2)', () => {
    it('calls setMessages([]) when Clear Chat button is clicked (AC1)', async () => {
      const user = userEvent.setup();
      const mockSetMessages = vi.fn();

      vi.mocked(useChat).mockReturnValue({
        ...createMockUseChatReturn({
          messages: [
            {
              id: 'msg-1',
              role: 'user',
              parts: [{ type: 'text', text: 'Hello' }],
            },
          ],
          sendMessage: mockSendMessage,
        }),
        setMessages: mockSetMessages,
      } as unknown as ReturnType<typeof useChat>);

      render(<ChatPanel workspaceId="test-workspace" />);

      const clearButton = screen.getByRole('button', { name: /clear chat/i });
      await user.click(clearButton);

      expect(mockSetMessages).toHaveBeenCalledWith([]);
    });

    it('calls setMessages([]) when Refresh Context button is clicked (AC2)', async () => {
      const user = userEvent.setup();
      const mockSetMessages = vi.fn();

      vi.mocked(useChat).mockReturnValue({
        ...createMockUseChatReturn({
          messages: [
            {
              id: 'msg-1',
              role: 'user',
              parts: [{ type: 'text', text: 'Hello' }],
            },
          ],
          sendMessage: mockSendMessage,
        }),
        setMessages: mockSetMessages,
      } as unknown as ReturnType<typeof useChat>);

      render(<ChatPanel workspaceId="test-workspace" />);

      const refreshButton = screen.getByRole('button', { name: /refresh context/i });
      await user.click(refreshButton);

      expect(mockSetMessages).toHaveBeenCalledWith([]);
    });

    it('disables Clear and Refresh buttons when loading', () => {
      vi.mocked(useChat).mockReturnValue(createMockUseChatReturn({
        status: 'streaming',
        sendMessage: mockSendMessage,
      }));

      render(<ChatPanel workspaceId="test-workspace" />);

      const clearButton = screen.getByRole('button', { name: /clear chat/i });
      const refreshButton = screen.getByRole('button', { name: /refresh context/i });

      expect(clearButton).toBeDisabled();
      expect(refreshButton).toBeDisabled();
    });
  });
});
