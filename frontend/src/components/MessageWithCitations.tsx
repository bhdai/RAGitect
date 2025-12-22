/**
 * Message content renderer with inline citation badges
 *
 * Renders LLM response text with [N] markers replaced by interactive
 * citation badges. Each badge shows a hover card with source preview.
 */

'use client';

import React, { memo } from 'react';
import { Streamdown } from 'streamdown';
import ReactMarkdown from 'react-markdown';
import { useTheme } from 'next-themes';
import { FileText } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from '@/components/ui/hover-card';
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area';
import type { CitationMap, CitationData } from '@/types/citation';
import { cn } from '@/lib/utils';

/**
 * Props for MessageWithCitations component
 */
export interface MessageWithCitationsProps {
  /** Raw text content with [N] citation markers */
  content: string;
  /** Map of citation IDs (cite-0, cite-1) to citation data */
  citations: CitationMap;
  /** Callback when user clicks a citation badge*/
  onCitationClick?: (citation: CitationData) => void;
}

/**
 * Simple markdown renderer for citation previews
 * Uses same approach as DocumentViewer - plain ReactMarkdown without code syntax highlighting
 */
function CitationPreviewMarkdown({ content }: { content: string }) {
  return (
    <ReactMarkdown
      components={{
        // Basic headings
        h1: (props) => <h1 className="text-sm font-bold mb-2 mt-3 first:mt-0 break-words" {...props} />,
        h2: (props) => <h2 className="text-sm font-bold mb-2 mt-3 break-words" {...props} />,
        h3: (props) => <h3 className="text-xs font-semibold mb-1 mt-2 break-words" {...props} />,
        // Lists
        ul: (props) => <ul className="list-disc pl-4 my-2 space-y-0.5 text-xs" {...props} />,
        ol: (props) => <ol className="list-decimal pl-4 my-2 space-y-0.5 text-xs" {...props} />,
        li: (props) => <li className="leading-relaxed break-words" {...props} />,
        // Paragraphs
        p: (props) => <p className="mb-2 text-xs leading-relaxed break-words" {...props} />,
        // Code - simple styling without syntax highlighting
        code: (props) => <code className="bg-muted px-1 py-0.5 rounded text-xs font-mono break-all" {...props} />,
        pre: (props) => <pre className="bg-muted p-2 rounded text-xs overflow-x-auto my-2" {...props} />,
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

/**
 * Citation badge component with hover card
 *
 * Displays citation number as a superscript badge. On hover, shows
 * a scrollable card with document name, relevance score, and full chunk content with markdown.
 */
function CitationBadge({
  citationIndex,
  citation,
  onCitationClick,
}: {
  citationIndex: number;
  citation: CitationData;
  onCitationClick?: (citation: CitationData) => void;
}) {
  const relevancePercent = Math.round(citation.similarity * 100);

  return (
    <HoverCard openDelay={200} closeDelay={100}>
      <HoverCardTrigger asChild>
        <button
          type="button"
          className={cn(
            'inline-flex items-center justify-center',
            'h-4 min-w-4 px-1 mx-0.5',
            'text-[10px] font-medium',
            'bg-primary/10 text-primary hover:bg-primary/20',
            'rounded-sm cursor-pointer',
            'align-super transition-colors',
            'focus:outline-none focus:ring-2 focus:ring-primary/50'
          )}
          onClick={() => onCitationClick?.(citation)}
          aria-label={`Citation ${citationIndex}: ${citation.title}`}
        >
          {citationIndex}
        </button>
      </HoverCardTrigger>
      <HoverCardContent
        className="w-96 max-w-lg p-0"
        side="top"
        align="start"
        sideOffset={5}
      >
        {/* Header with document icon and name */}
        <div className="flex items-start gap-2 p-3 border-b bg-muted/30">
          <FileText className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium truncate" title={citation.title}>
              {citation.title}
            </p>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Badge variant="secondary" className="h-4 text-[10px] px-1.5">
                {relevancePercent}% relevant
              </Badge>
            </div>
          </div>
        </div>

        {/* Content with markdown rendering, vertical and horizontal scroll */}
        {citation.preview && (
          <ScrollArea className="h-64 w-full">
            <div className="p-3 break-words overflow-wrap-anywhere">
              <CitationPreviewMarkdown content={citation.preview} />
            </div>
            <ScrollBar orientation="horizontal" />
          </ScrollArea>
        )}
      </HoverCardContent>
    </HoverCard>
  );
}

/**
 * Process text to replace [N] markers with citation badges
 */
function processTextWithCitations(
  text: string,
  citations: CitationMap,
  onCitationClick?: (citation: CitationData) => void
): React.ReactNode[] {
  const result: React.ReactNode[] = [];
  const pattern = /\[(\d+)\]/g;
  let lastIndex = 0;
  let match;
  let keyIndex = 0;

  while ((match = pattern.exec(text)) !== null) {
    // Add text before this match
    if (match.index > lastIndex) {
      result.push(text.slice(lastIndex, match.index));
    }

    const citationIndex = parseInt(match[1], 10);
    const citationId = `cite-${citationIndex}`;
    const citation = citations[citationId];

    if (citation) {
      result.push(
        <CitationBadge
          key={`cite-${keyIndex++}`}
          citationIndex={citationIndex}
          citation={citation}
          onCitationClick={onCitationClick}
        />
      );
    } else {
      // Graceful degradation for invalid citations
      result.push(
        <span key={`invalid-${keyIndex++}`} className="text-muted-foreground">
          [{citationIndex}]
        </span>
      );
    }

    lastIndex = pattern.lastIndex;
  }

  // Add remaining text
  if (lastIndex < text.length) {
    result.push(text.slice(lastIndex));
  }

  return result;
}

/**
 * Renders message content with inline citation badges
 *
 * Uses Streamdown for rich markdown rendering with Shiki syntax highlighting,
 * code copy buttons, and beautiful formatting. Citations are injected inline
 * using Streamdown's components prop.
 *
 * @example
 * ```tsx
 * <MessageWithCitations
 *   content="Python is a powerful language [0] that supports multiple paradigms [1]."
 *   citations={{
 *     'cite-0': { id: 'cite-0', title: 'intro.pdf', similarity: 0.95, preview: '...' },
 *     'cite-1': { id: 'cite-1', title: 'advanced.pdf', similarity: 0.87, preview: '...' },
 *   }}
 * />
 * ```
 */
export const MessageWithCitations = memo(function MessageWithCitations({
  content,
  citations,
  onCitationClick,
}: MessageWithCitationsProps) {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === 'dark';

  // Check if we have any citations to render
  const hasCitations = Object.keys(citations).length > 0;

  // Process children to replace text containing [N] with citation badges
  const processChildren = React.useCallback((children: React.ReactNode): React.ReactNode => {
    if (typeof children === 'string') {
      return processTextWithCitations(children, citations, onCitationClick);
    }
    if (Array.isArray(children)) {
      return children.map((child, i) => {
        if (typeof child === 'string') {
          const processed = processTextWithCitations(child, citations, onCitationClick);
          return processed.length === 1 ? processed[0] : <span key={i}>{processed}</span>;
        }
        return child;
      });
    }
    return children;
  }, [citations, onCitationClick]);

  // Custom components for Streamdown that inject citation badges
  const components = React.useMemo(() => hasCitations ? {
    p: ({ children, ...props }: React.ComponentPropsWithoutRef<'p'>) => {
      const processedChildren = processChildren(children);
      return <p {...props}>{processedChildren}</p>;
    },
    li: ({ children, ...props }: React.ComponentPropsWithoutRef<'li'>) => {
      const processedChildren = processChildren(children);
      return <li {...props}>{processedChildren}</li>;
    },
    strong: ({ children, ...props }: React.ComponentPropsWithoutRef<'strong'>) => {
      const processedChildren = processChildren(children);
      return <strong {...props}>{processedChildren}</strong>;
    },
    em: ({ children, ...props }: React.ComponentPropsWithoutRef<'em'>) => {
      const processedChildren = processChildren(children);
      return <em {...props}>{processedChildren}</em>;
    },
  } : undefined, [hasCitations, processChildren]);

  return (
    <Streamdown
      className="size-full [&>*:first-child]:mt-0 [&>*:last-child]:mb-0"
      shikiTheme={isDark ? ['github-dark', 'github-dark'] : ['github-light', 'github-light']}
      components={components}
    >
      {content}
    </Streamdown>
  );
});
