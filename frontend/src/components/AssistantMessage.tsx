'use client';

import { memo, useMemo } from 'react';
import type { UIMessage } from 'ai';
import { MessageWithCitations } from '@/components/MessageWithCitations';
import { buildCitationMap, type CitationData } from '@/types/citation';

interface AssistantMessageProps {
  message: UIMessage;
  onCitationClick?: (citation: CitationData) => void;
}

export const AssistantMessage = memo(function AssistantMessage({
  message,
  onCitationClick,
}: AssistantMessageProps) {
  const content = useMemo(
    () =>
      message.parts
        ?.filter((part): part is { type: 'text'; text: string } => part.type === 'text')
        .map((part) => part.text)
        .join('') || '',
    [message.parts]
  );

  const citations = useMemo(() => buildCitationMap(message.parts), [message.parts]);

  return (
    <div className="w-full">
      <MessageWithCitations
        content={content}
        citations={citations}
        onCitationClick={onCitationClick}
      />
    </div>
  );
});
