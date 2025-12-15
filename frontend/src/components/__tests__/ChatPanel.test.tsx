/**
 * Tests for ChatPanel component
 *
 * Story 3.0: Streaming Infrastructure - AC3
 * Story 3.1: Natural Language Querying - AC1, AC4
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ChatPanel } from '../ChatPanel';

// Mock the useChat hook
vi.mock('@/hooks/useChat', () => ({
  useChat: vi.fn(),
}));

import { useChat } from '@/hooks/useChat';

describe('ChatPanel', () => {
  const mockSendMessage = vi.fn();
  const mockClearMessages = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useChat).mockReturnValue({
      messages: [],
      isLoading: false,
      error: null,
      sendMessage: mockSendMessage,
      clearMessages: mockClearMessages,
    });
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
    vi.mocked(useChat).mockReturnValue({
      messages: [{ role: 'user', content: 'Hello there!' }],
      isLoading: false,
      error: null,
      sendMessage: mockSendMessage,
      clearMessages: mockClearMessages,
    });

    render(<ChatPanel workspaceId="test-workspace" />);

    expect(screen.getByText('Hello there!')).toBeInTheDocument();
  });

  it('renders assistant messages', () => {
    vi.mocked(useChat).mockReturnValue({
      messages: [
        { role: 'user', content: 'What is Python?' },
        { role: 'assistant', content: 'Python is a programming language.' },
      ],
      isLoading: false,
      error: null,
      sendMessage: mockSendMessage,
      clearMessages: mockClearMessages,
    });

    render(<ChatPanel workspaceId="test-workspace" />);

    expect(screen.getByText('What is Python?')).toBeInTheDocument();
    expect(
      screen.getByText('Python is a programming language.')
    ).toBeInTheDocument();
  });

  it('shows loading indicator when processing', () => {
    vi.mocked(useChat).mockReturnValue({
      messages: [
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: '' },
      ],
      isLoading: true,
      error: null,
      sendMessage: mockSendMessage,
      clearMessages: mockClearMessages,
    });

    render(<ChatPanel workspaceId="test-workspace" />);

    expect(screen.getByText(/thinking/i)).toBeInTheDocument();
  });

  it('shows error message when error occurs', () => {
    vi.mocked(useChat).mockReturnValue({
      messages: [],
      isLoading: false,
      error: 'Connection failed',
      sendMessage: mockSendMessage,
      clearMessages: mockClearMessages,
    });

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

    expect(mockSendMessage).toHaveBeenCalledWith('What is RAG?');
  });

  it('sends message when Enter key is pressed', async () => {
    const user = userEvent.setup();
    mockSendMessage.mockResolvedValue(undefined);

    render(<ChatPanel workspaceId="test-workspace" />);

    const input = screen.getByPlaceholderText(/ask a question/i);
    await user.type(input, 'Hello{Enter}');

    expect(mockSendMessage).toHaveBeenCalledWith('Hello');
  });

  it('does not send empty messages', async () => {
    const user = userEvent.setup();

    render(<ChatPanel workspaceId="test-workspace" />);

    const submitButton = screen.getByRole('button');
    await user.click(submitButton);

    expect(mockSendMessage).not.toHaveBeenCalled();
  });

  it('disables input when loading', () => {
    vi.mocked(useChat).mockReturnValue({
      messages: [],
      isLoading: true,
      error: null,
      sendMessage: mockSendMessage,
      clearMessages: mockClearMessages,
    });

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
