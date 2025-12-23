/**
 * Message action buttons component
 *
 * Provides copy button with citation stripping for assistant messages.
 * AC3: Copy button strips [cite: N] markers before copying
 * AC4: Visual feedback with checkmark for 2 seconds after copy
 */

'use client';

import React, { memo } from 'react';
import { Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useCopyToClipboard } from '@/hooks/useCopyToClipboard';
import { stripCitations } from '@/lib/citations';
import { cn } from '@/lib/utils';

export interface MessageActionsProps {
  /** Raw content with [cite: N] markers to copy */
  rawContent: string;
  /** Optional callback after successful copy */
  onCopy?: () => void;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Action buttons for assistant messages
 *
 * Currently supports:
 * - Copy to clipboard (strips citations for clean paste)
 *
 * Future extensibility:
 * - Save to Note
 *
 * @example
 * ```tsx
 * <MessageActions
 *   rawContent="Python is powerful [cite: 0]."
 *   onCopy={() => toast.success('Copied!')}
 * />
 * ```
 */
export const MessageActions = memo(function MessageActions({
  rawContent,
  onCopy,
  className,
}: MessageActionsProps) {
  const { copied, copy } = useCopyToClipboard();

  const handleCopy = async () => {
    // Strip citations before copying (AC3)
    const cleanText = stripCitations(rawContent);
    const success = await copy(cleanText);

    if (success && onCopy) {
      onCopy();
    }
  };

  return (
    <div className={cn('flex items-center gap-1', className)}>
      <Button
        variant="ghost"
        size="icon"
        className="h-7 w-7"
        onClick={handleCopy}
        aria-label={copied ? 'Copied to clipboard' : 'Copy message'}
      >
        {copied ? (
          <Check
            className="h-4 w-4 text-green-500"
            data-testid="check-icon"
          />
        ) : (
          <Copy
            className="h-4 w-4 text-muted-foreground"
            data-testid="copy-icon"
          />
        )}
      </Button>
    </div>
  );
});
