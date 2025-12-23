/**
 * Tests for MessageActions component
 *
 * AC3/AC4: Copy button with citation stripping and visual feedback
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { MessageActions } from '../MessageActions';

// Mock the useCopyToClipboard hook
const mockCopy = vi.fn();
vi.mock('@/hooks/useCopyToClipboard', () => ({
  useCopyToClipboard: () => ({
    copied: false,
    copy: mockCopy,
  }),
}));

describe('MessageActions', () => {
  beforeEach(() => {
    mockCopy.mockClear();
    mockCopy.mockResolvedValue(true);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('renders copy button', () => {
    render(<MessageActions rawContent="Hello world" />);

    const button = screen.getByRole('button', { name: /copy/i });
    expect(button).toBeInTheDocument();
  });

  it('strips citations before copying (AC3)', async () => {
    const user = userEvent.setup();

    render(
      <MessageActions rawContent="Python is powerful [cite: 1] and versatile [cite: 2]." />
    );

    const button = screen.getByRole('button', { name: /copy/i });
    await user.click(button);

    // Should copy WITHOUT citation markers
    expect(mockCopy).toHaveBeenCalledWith('Python is powerful and versatile.');
  });

  it('calls onCopy callback when provided', async () => {
    const user = userEvent.setup();
    const onCopy = vi.fn();

    render(<MessageActions rawContent="Hello world" onCopy={onCopy} />);

    const button = screen.getByRole('button', { name: /copy/i });
    await user.click(button);

    await waitFor(() => {
      expect(onCopy).toHaveBeenCalled();
    });
  });

  it('handles empty content', async () => {
    const user = userEvent.setup();

    render(<MessageActions rawContent="" />);

    const button = screen.getByRole('button', { name: /copy/i });
    await user.click(button);

    expect(mockCopy).toHaveBeenCalledWith('');
  });

  it('handles content with only citations', async () => {
    const user = userEvent.setup();

    render(<MessageActions rawContent="[cite: 1][cite: 2]" />);

    const button = screen.getByRole('button', { name: /copy/i });
    await user.click(button);

    expect(mockCopy).toHaveBeenCalledWith('');
  });
});

describe('MessageActions visual feedback', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it('shows check icon after successful copy (AC4)', async () => {
    // Re-mock with copied = true to test visual state
    vi.doMock('@/hooks/useCopyToClipboard', () => ({
      useCopyToClipboard: () => ({
        copied: true,
        copy: vi.fn().mockResolvedValue(true),
      }),
    }));

    // Re-import to get mocked version
    const { MessageActions: MessageActionsCopied } = await import(
      '../MessageActions'
    );

    render(<MessageActionsCopied rawContent="Hello world" />);

    // Shows check icon when copied
    expect(screen.queryByTestId('check-icon')).toBeInTheDocument();
    expect(screen.queryByTestId('copy-icon')).not.toBeInTheDocument();
  });

  it('shows copy icon initially', async () => {
    vi.doMock('@/hooks/useCopyToClipboard', () => ({
      useCopyToClipboard: () => ({
        copied: false,
        copy: vi.fn().mockResolvedValue(true),
      }),
    }));

    const { MessageActions: MessageActionsNotCopied } = await import(
      '../MessageActions'
    );

    render(<MessageActionsNotCopied rawContent="Hello world" />);

    expect(screen.queryByTestId('copy-icon')).toBeInTheDocument();
    expect(screen.queryByTestId('check-icon')).not.toBeInTheDocument();
  });
});
