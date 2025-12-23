/**
 * Tests for MessageWithCitations component
 *
 * Tests citation parsing, badge rendering, and graceful degradation.
 */

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { MessageWithCitations } from '../MessageWithCitations';
import type { CitationMap } from '@/types/citation';

describe('MessageWithCitations', () => {
  const mockCitations: CitationMap = {
    'cite-1': {
      id: 'cite-1',
      title: 'intro.pdf',
      mediaType: 'text/plain',
      chunkIndex: 0,
      similarity: 0.95,
      preview: 'Python is a powerful programming language used worldwide...',
      documentId: 'doc-uuid-1',
    },
    'cite-2': {
      id: 'cite-2',
      title: 'advanced.pdf',
      mediaType: 'text/plain',
      chunkIndex: 5,
      similarity: 0.87,
      preview: 'Python supports multiple programming paradigms including OOP...',
      documentId: 'doc-uuid-2',
    },
  };

  describe('citation parsing', () => {
    it('renders text without citations normally', () => {
      render(
        <MessageWithCitations
          content="Python is a powerful language."
          citations={{}}
        />
      );

      expect(screen.getByText('Python is a powerful language.')).toBeInTheDocument();
    });

    it('renders citation badges for [cite: N] markers', () => {
      render(
        <MessageWithCitations
          content="Python is powerful[cite: 1] and versatile[cite: 2]."
          citations={mockCitations}
        />
      );

      // Should render citation buttons
      const buttons = screen.getAllByRole('button');
      expect(buttons).toHaveLength(2);
      expect(buttons[0]).toHaveTextContent('1');
      expect(buttons[1]).toHaveTextContent('2');
    });

    it('does not match old bare [N] format (AC5)', () => {
      render(
        <MessageWithCitations
          content="Python is powerful[0] and versatile[1]."
          citations={mockCitations}
        />
      );

      // Should NOT render citation buttons - old format should not match
      const buttons = screen.queryAllByRole('button');
      expect(buttons).toHaveLength(0);
    });

    it('handles multiple citations in sequence', () => {
      render(
        <MessageWithCitations
          content="This fact is supported by multiple sources[cite: 1][cite: 2]."
          citations={mockCitations}
        />
      );

      const buttons = screen.getAllByRole('button');
      expect(buttons).toHaveLength(2);
    });

    it('handles citation at start of text', () => {
      render(
        <MessageWithCitations
          content="[cite: 1] Python is great."
          citations={mockCitations}
        />
      );

      const button = screen.getByRole('button');
      expect(button).toHaveTextContent('1');
    });

    it('handles citation at end of text', () => {
      render(
        <MessageWithCitations
          content="Python is great [cite: 1]"
          citations={mockCitations}
        />
      );

      const button = screen.getByRole('button');
      expect(button).toHaveTextContent('1');
    });

    it('handles citation with space variations [cite:1] and [cite: 1]', () => {
      render(
        <MessageWithCitations
          content="Python[cite:1] and JavaScript[cite: 1]."
          citations={mockCitations}
        />
      );

      const buttons = screen.getAllByRole('button');
      expect(buttons).toHaveLength(2);
    });
  });

  describe('citation badge interaction', () => {
    it('has correct aria-label for accessibility', () => {
      render(
        <MessageWithCitations
          content="Python is powerful[cite: 1]."
          citations={mockCitations}
        />
      );

      const button = screen.getByRole('button');
      expect(button).toHaveAttribute(
        'aria-label',
        'Citation 1: intro.pdf'
      );
    });

    it('calls onCitationClick when clicked', async () => {
      const user = userEvent.setup();
      const handleClick = vi.fn();

      render(
        <MessageWithCitations
          content="Python is powerful[cite: 1]."
          citations={mockCitations}
          onCitationClick={handleClick}
        />
      );

      const button = screen.getByRole('button');
      await user.click(button);

      expect(handleClick).toHaveBeenCalledWith(mockCitations['cite-1']);
    });
  });

  describe('hover card content', () => {
    it('shows document title on hover', async () => {
      const user = userEvent.setup();

      render(
        <MessageWithCitations
          content="Python is powerful[cite: 1]."
          citations={mockCitations}
        />
      );

      const button = screen.getByRole('button');
      await user.hover(button);

      // Wait for hover card to appear
      const title = await screen.findByText('intro.pdf');
      expect(title).toBeInTheDocument();
    });

    it('shows relevance score as percentage', async () => {
      const user = userEvent.setup();

      render(
        <MessageWithCitations
          content="Python is powerful[cite: 1]."
          citations={mockCitations}
        />
      );

      const button = screen.getByRole('button');
      await user.hover(button);

      // 95% relevance from mockCitations
      const score = await screen.findByText('95% relevant');
      expect(score).toBeInTheDocument();
    });

    it('shows content preview in quotes', async () => {
      const user = userEvent.setup();

      render(
        <MessageWithCitations
          content="Python is powerful[cite: 1]."
          citations={mockCitations}
        />
      );

      const button = screen.getByRole('button');
      await user.hover(button);

      // Preview text should be in the hover card (quotes are rendered as HTML entities)
      const preview = await screen.findByText(
        /Python is a powerful programming language used worldwide/
      );
      expect(preview).toBeInTheDocument();
    });
  });

  describe('graceful degradation', () => {
    it('renders invalid citation as plain text', () => {
      render(
        <MessageWithCitations
          content="Python is powerful[cite: 99]."
          citations={mockCitations}
        />
      );

      // Should not be a button
      const buttons = screen.queryAllByRole('button');
      expect(buttons).toHaveLength(0);

      // Should show [99] as plain text
      expect(screen.getByText('[99]')).toBeInTheDocument();
    });

    it('handles mixed valid and invalid citations', () => {
      render(
        <MessageWithCitations
          content="Fact one[cite: 1] and fact two[cite: 99]."
          citations={mockCitations}
        />
      );

      // Should have one button for valid citation
      const buttons = screen.getAllByRole('button');
      expect(buttons).toHaveLength(1);
      expect(buttons[0]).toHaveTextContent('1');

      // Invalid should be plain text
      expect(screen.getByText('[99]')).toBeInTheDocument();
    });

    it('handles empty citations map', () => {
      render(
        <MessageWithCitations
          content="Python is powerful[cite: 1]."
          citations={{}}
        />
      );

      // Should just render the text without parsing citations
      expect(screen.getByText('Python is powerful[cite: 1].')).toBeInTheDocument();
    });
  });

  describe('zero citations case', () => {
    it('renders normally when no citation markers in text', () => {
      render(
        <MessageWithCitations
          content="2 plus 2 equals 4."
          citations={mockCitations}
        />
      );

      // Text should be rendered (uses react-markdown when hasCitations=true)
      expect(screen.getByText('2 plus 2 equals 4.')).toBeInTheDocument();
      expect(screen.queryAllByRole('button')).toHaveLength(0);
    });
  });
});
