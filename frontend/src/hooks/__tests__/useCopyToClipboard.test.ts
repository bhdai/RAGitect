/**
 * Tests for useCopyToClipboard hook
 *
 * AC3/AC4: Hook for copying text with visual feedback
 */

import { renderHook, act, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { useCopyToClipboard } from '../../hooks/useCopyToClipboard';

describe('useCopyToClipboard', () => {
  const mockClipboard = {
    writeText: vi.fn(),
  };

  beforeEach(() => {
    vi.useFakeTimers();
    Object.assign(navigator, {
      clipboard: mockClipboard,
    });
    mockClipboard.writeText.mockResolvedValue(undefined);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it('initializes with copied = false', () => {
    const { result } = renderHook(() => useCopyToClipboard());

    expect(result.current.copied).toBe(false);
  });

  it('copies text to clipboard and sets copied = true', async () => {
    const { result } = renderHook(() => useCopyToClipboard());

    await act(async () => {
      const success = await result.current.copy('Hello world');
      expect(success).toBe(true);
    });

    expect(mockClipboard.writeText).toHaveBeenCalledWith('Hello world');
    expect(result.current.copied).toBe(true);
  });

  it('resets copied to false after 2 seconds', async () => {
    const { result } = renderHook(() => useCopyToClipboard());

    await act(async () => {
      await result.current.copy('Hello world');
    });

    expect(result.current.copied).toBe(true);

    act(() => {
      vi.advanceTimersByTime(2000);
    });

    expect(result.current.copied).toBe(false);
  });

  it('accepts custom timeout duration', async () => {
    const { result } = renderHook(() => useCopyToClipboard({ timeout: 3000 }));

    await act(async () => {
      await result.current.copy('Hello world');
    });

    expect(result.current.copied).toBe(true);

    act(() => {
      vi.advanceTimersByTime(2000);
    });

    // Still copied after 2 seconds (timeout is 3 seconds)
    expect(result.current.copied).toBe(true);

    act(() => {
      vi.advanceTimersByTime(1000);
    });

    // Now it should be reset
    expect(result.current.copied).toBe(false);
  });

  it('returns false when clipboard write fails', async () => {
    mockClipboard.writeText.mockRejectedValueOnce(new Error('Permission denied'));

    const { result } = renderHook(() => useCopyToClipboard());

    await act(async () => {
      const success = await result.current.copy('Hello world');
      expect(success).toBe(false);
    });

    expect(result.current.copied).toBe(false);
  });

  it('cancels previous timeout when copying again', async () => {
    const { result } = renderHook(() => useCopyToClipboard());

    // First copy
    await act(async () => {
      await result.current.copy('First');
    });

    expect(result.current.copied).toBe(true);

    // Advance 1.5 seconds
    act(() => {
      vi.advanceTimersByTime(1500);
    });

    // Copy again before timeout
    await act(async () => {
      await result.current.copy('Second');
    });

    expect(result.current.copied).toBe(true);

    // Advance 1.5 seconds (would have reset first copy)
    act(() => {
      vi.advanceTimersByTime(1500);
    });

    // Still copied because new timeout
    expect(result.current.copied).toBe(true);

    // Advance remaining 0.5 seconds
    act(() => {
      vi.advanceTimersByTime(500);
    });

    expect(result.current.copied).toBe(false);
  });

  it('handles empty string', async () => {
    const { result } = renderHook(() => useCopyToClipboard());

    await act(async () => {
      const success = await result.current.copy('');
      expect(success).toBe(true);
    });

    expect(mockClipboard.writeText).toHaveBeenCalledWith('');
  });
});
