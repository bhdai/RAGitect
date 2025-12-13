/**
 * DocumentViewer Unit Tests
 * 
 * Tests document content rendering and panel behavior.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DocumentViewer } from '../DocumentViewer';

// Mock the documents API
vi.mock('@/lib/documents', () => ({
  getDocument: vi.fn(),
}));

// Mock react-markdown
vi.mock('react-markdown', () => ({
  default: ({ children }: { children: string }) => <div data-testid="markdown-content">{children}</div>,
}));

// Reset mocks before each test
beforeEach(() => {
  vi.clearAllMocks();
});

describe('DocumentViewer', () => {
  it('renders document content when documentId is provided', async () => {
    const { getDocument } = await import('@/lib/documents');
    const mockDoc = {
      id: '1',
      fileName: 'test.pdf',
      fileType: 'pdf',
      status: 'ready',
      processedContent: '# Test Document\n\nThis is the content.',
      summary: null,
      createdAt: '2025-12-01T10:00:00Z',
    };
    vi.mocked(getDocument).mockResolvedValue(mockDoc);

    render(<DocumentViewer documentId="1" onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText('test.pdf')).toBeInTheDocument();
      expect(screen.getByTestId('markdown-content')).toHaveTextContent('# Test Document');
    });
  });

  it('renders nothing when documentId is null', () => {
    const { container } = render(<DocumentViewer documentId={null} onClose={vi.fn()} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('displays loading state during fetch', async () => {
    const { getDocument } = await import('@/lib/documents');
    vi.mocked(getDocument).mockReturnValue(new Promise(() => {})); // Never resolves

    render(<DocumentViewer documentId="1" onClose={vi.fn()} />);

    expect(screen.getByText(/loading document/i)).toBeInTheDocument();
  });

  it('calls onClose when close button is clicked', async () => {
    const user = userEvent.setup();
    const { getDocument } = await import('@/lib/documents');
    const mockDoc = {
      id: '1',
      fileName: 'test.pdf',
      fileType: 'pdf',
      status: 'ready',
      processedContent: 'Content',
      summary: null,
      createdAt: '2025-12-01T10:00:00Z',
    };
    vi.mocked(getDocument).mockResolvedValue(mockDoc);

    const handleClose = vi.fn();
    render(<DocumentViewer documentId="1" onClose={handleClose} />);

    await waitFor(() => {
      expect(screen.getByText('test.pdf')).toBeInTheDocument();
    });

    const closeButton = screen.getByLabelText(/close/i);
    await user.click(closeButton);

    expect(handleClose).toHaveBeenCalled();
  });

  it('displays empty content message when processedContent is null', async () => {
    const { getDocument } = await import('@/lib/documents');
    const mockDoc = {
      id: '1',
      fileName: 'unprocessed.pdf',
      fileType: 'pdf',
      status: 'uploaded',
      processedContent: null,
      summary: null,
      createdAt: '2025-12-01T10:00:00Z',
    };
    vi.mocked(getDocument).mockResolvedValue(mockDoc);

    render(<DocumentViewer documentId="1" onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText(/no content available/i)).toBeInTheDocument();
    });
  });

  it('fetches new document when documentId changes', async () => {
    const { getDocument } = await import('@/lib/documents');
    const mockDoc1 = {
      id: '1',
      fileName: 'doc1.pdf',
      fileType: 'pdf',
      status: 'ready',
      processedContent: 'Content 1',
      summary: null,
      createdAt: '2025-12-01T10:00:00Z',
    };
    const mockDoc2 = {
      id: '2',
      fileName: 'doc2.pdf',
      fileType: 'pdf',
      status: 'ready',
      processedContent: 'Content 2',
      summary: null,
      createdAt: '2025-12-02T10:00:00Z',
    };
    vi.mocked(getDocument).mockResolvedValueOnce(mockDoc1).mockResolvedValueOnce(mockDoc2);

    const { rerender } = render(<DocumentViewer documentId="1" onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText('doc1.pdf')).toBeInTheDocument();
    });

    rerender(<DocumentViewer documentId="2" onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText('doc2.pdf')).toBeInTheDocument();
    });
  });
});
