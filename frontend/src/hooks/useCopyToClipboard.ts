/**
 * Hook for copying text to clipboard with visual feedback
 *
 * AC3/AC4: Provides copy functionality with temporary "copied" state
 * that resets after a configurable timeout (default 2 seconds).
 */

'use client';

import { useState, useCallback, useRef } from 'react';

export interface UseCopyToClipboardOptions {
  /** Duration in ms to show "copied" state (default: 2000) */
  timeout?: number;
}

export interface UseCopyToClipboardReturn {
  /** Whether text was recently copied */
  copied: boolean;
  /** Copy text to clipboard, returns success boolean */
  copy: (text: string) => Promise<boolean>;
}

/**
 * Hook for copying text to clipboard with visual feedback
 *
 * @param options - Configuration options
 * @returns Object with copied state and copy function
 *
 * @example
 * ```tsx
 * const { copied, copy } = useCopyToClipboard();
 *
 * <button onClick={() => copy(text)}>
 *   {copied ? <Check /> : <Copy />}
 * </button>
 * ```
 */
export function useCopyToClipboard(
  options: UseCopyToClipboardOptions = {}
): UseCopyToClipboardReturn {
  const { timeout = 2000 } = options;
  const [copied, setCopied] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const copy = useCallback(
    async (text: string): Promise<boolean> => {
      // Cancel any existing timeout
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }

      try {
        await navigator.clipboard.writeText(text);
        setCopied(true);

        // Reset after timeout
        timeoutRef.current = setTimeout(() => {
          setCopied(false);
          timeoutRef.current = null;
        }, timeout);

        return true;
      } catch {
        setCopied(false);
        return false;
      }
    },
    [timeout]
  );

  return { copied, copy };
}
