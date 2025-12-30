/**
 * Citation Deep-Dive Integration Tests
 *
 * Tests the citation click flow:
 * - AC1: Clicking a citation badge opens the document viewer
 * - AC2: The viewer loads the correct document
 * - AC6: Edge cases handled gracefully
 *
 * Note: Highlighting (AC3/4/5) has been deferred to a future implementation
 * due to technical complexity with markdown rendering and text anchoring.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MessageWithCitations } from '../MessageWithCitations';
import type { CitationMap, CitationData } from '@/types/citation';

vi.mock('react-markdown', () => ({
  default: ({ children }: { children: string }) => <div data-testid="markdown-content">{children}</div>,
}));

describe('Citation Deep-Dive Integration', () => {
  const mockCitations: CitationMap = {
    'cite-0': {
      id: 'cite-0',
      title: 'architecture.md',
      mediaType: 'text/plain',
      chunkIndex: 2,
      similarity: 0.95,
      preview: 'The system uses a **microservices** architecture with REST APIs.',
      documentId: 'doc-uuid-123',
    },
    'cite-1': {
      id: 'cite-1',
      title: 'deployment.md',
      mediaType: 'text/plain',
      chunkIndex: 5,
      similarity: 0.87,
      preview: 'Docker containers are deployed via Kubernetes.',
      documentId: 'doc-uuid-456',
    },
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('AC1: Clicking citation opens DocumentViewer', () => {
    it('calls onCitationClick with full citation data when badge clicked', async () => {
      const user = userEvent.setup();
      const handleCitationClick = vi.fn();

      render(
        <MessageWithCitations
          content="The architecture is well documented [cite: 1]."
          citations={mockCitations}
          onCitationClick={handleCitationClick}
        />
      );

      const citationBadge = screen.getByRole('button', { name: /Citation 1/i });
      await user.click(citationBadge);

      expect(handleCitationClick).toHaveBeenCalledWith(mockCitations['cite-0']);
    });
  });

  describe('AC2: Viewer loads correct document by documentId', () => {
    it('passes correct documentId from citation to viewer', async () => {
      const user = userEvent.setup();
      let receivedCitation: CitationData | null = null;

      const handleCitationClick = (citation: CitationData) => {
        receivedCitation = citation;
      };

      render(
        <MessageWithCitations
          content="Check the deployment guide [cite: 2]."
          citations={mockCitations}
          onCitationClick={handleCitationClick}
        />
      );

      const citationBadge = screen.getByRole('button', { name: /Citation 2/i });
      await user.click(citationBadge);

      expect(receivedCitation).not.toBeNull();
      expect(receivedCitation!.documentId).toBe('doc-uuid-456');
      expect(receivedCitation!.id).toBe('cite-1');
    });
  });

  describe('AC6: Edge cases', () => {
    it('handles missing documentId gracefully', async () => {
      const user = userEvent.setup();
      const handleCitationClick = vi.fn();

      const citationWithoutDocId: CitationMap = {
        'cite-0': {
          id: 'cite-0',
          title: 'orphan.md',
          mediaType: 'text/plain',
          chunkIndex: 0,
          similarity: 0.9,
          preview: 'Some preview text',
          documentId: '', // Empty documentId
        },
      };

      render(
        <MessageWithCitations
          content="Check this [cite: 1]."
          citations={citationWithoutDocId}
          onCitationClick={handleCitationClick}
        />
      );

      const citationBadge = screen.getByRole('button', { name: /Citation 1/i });
      await user.click(citationBadge);

      // The handler is called - workspace page validates and shows error
      expect(handleCitationClick).toHaveBeenCalledWith(citationWithoutDocId['cite-0']);
    });
  });
});
