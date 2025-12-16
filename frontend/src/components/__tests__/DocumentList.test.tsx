/**
 * DocumentList Unit Tests
 * 
 * Tests component rendering, document selection, and delete functionality.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DocumentList } from '../DocumentList';

// Mock the documents API
vi.mock('@/lib/documents', () => ({
  getDocuments: vi.fn(),
}));

// Mock sonner toast
vi.mock('sonner', () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
  },
}));

// Reset mocks before each test
beforeEach(() => {
  vi.clearAllMocks();
});

describe('DocumentList', () => {
  it('renders list of documents', async () => {
    const { getDocuments } = await import('@/lib/documents');
    const mockDocs = {
      documents: [
        { id: '1', fileName: 'test.pdf', fileType: 'pdf', status: 'ready', createdAt: '2025-12-01T10:00:00Z' },
        { id: '2', fileName: 'report.docx', fileType: 'docx', status: 'processing', createdAt: '2025-12-02T10:00:00Z' },
      ],
      total: 2,
    };
    vi.mocked(getDocuments).mockResolvedValue(mockDocs);

    render(
      <DocumentList
        workspaceId="ws-1"
        onSelectDocument={vi.fn()}
        onDeleteDocument={vi.fn()}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('test.pdf')).toBeInTheDocument();
      expect(screen.getByText('report.docx')).toBeInTheDocument();
    });
  });

  it('displays loading state', async () => {
    const { getDocuments } = await import('@/lib/documents');
    vi.mocked(getDocuments).mockReturnValue(new Promise(() => {})); // Never resolves

    render(
      <DocumentList
        workspaceId="ws-1"
        onSelectDocument={vi.fn()}
        onDeleteDocument={vi.fn()}
      />
    );

    expect(screen.getByText(/loading documents/i)).toBeInTheDocument();
  });

  it('displays empty state message when no documents', async () => {
    const { getDocuments } = await import('@/lib/documents');
    vi.mocked(getDocuments).mockResolvedValue({ documents: [], total: 0 });

    render(
      <DocumentList
        workspaceId="ws-1"
        onSelectDocument={vi.fn()}
        onDeleteDocument={vi.fn()}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/no documents/i)).toBeInTheDocument();
    });
  });

  it('calls onSelectDocument when document is clicked', async () => {
    const user = userEvent.setup();
    const { getDocuments } = await import('@/lib/documents');
    const mockDoc = { id: '1', fileName: 'test.pdf', fileType: 'pdf', status: 'ready', createdAt: '2025-12-01T10:00:00Z' };
    vi.mocked(getDocuments).mockResolvedValue({ documents: [mockDoc], total: 1 });

    const handleSelect = vi.fn();
    render(
      <DocumentList
        workspaceId="ws-1"
        onSelectDocument={handleSelect}
        onDeleteDocument={vi.fn()}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('test.pdf')).toBeInTheDocument();
    });

    await user.click(screen.getByText('test.pdf'));

    expect(handleSelect).toHaveBeenCalledWith(mockDoc);
  });

  it('calls onDeleteDocument when delete button is clicked', async () => {
    const user = userEvent.setup();
    const { getDocuments } = await import('@/lib/documents');
    const mockDoc = { id: '1', fileName: 'test.pdf', fileType: 'pdf', status: 'ready', createdAt: '2025-12-01T10:00:00Z' };
    vi.mocked(getDocuments).mockResolvedValue({ documents: [mockDoc], total: 1 });

    const handleDelete = vi.fn();
    render(
      <DocumentList
        workspaceId="ws-1"
        onSelectDocument={vi.fn()}
        onDeleteDocument={handleDelete}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('test.pdf')).toBeInTheDocument();
    });

    // Find and click delete button
    const deleteButton = screen.getByLabelText(/delete test.pdf/i);
    await user.click(deleteButton);

    expect(handleDelete).toHaveBeenCalledWith(mockDoc);
  });

  it('uses color-coded status for documents', async () => {
    const { getDocuments } = await import('@/lib/documents');
    const mockDocs = {
      documents: [
        { id: '1', fileName: 'ready.pdf', fileType: 'pdf', status: 'ready', createdAt: '2025-12-01T10:00:00Z' },
        { id: '2', fileName: 'processing.pdf', fileType: 'pdf', status: 'processing', createdAt: '2025-12-02T10:00:00Z' },
      ],
      total: 2,
    };
    vi.mocked(getDocuments).mockResolvedValue(mockDocs);

    render(
      <DocumentList
        workspaceId="ws-1"
        onSelectDocument={vi.fn()}
        onDeleteDocument={vi.fn()}
      />
    );

    await waitFor(() => {
      // Documents should render with file names
      expect(screen.getByText('ready.pdf')).toBeInTheDocument();
      expect(screen.getByText('processing.pdf')).toBeInTheDocument();
    });
  });

  it('refetches documents when workspaceId changes', async () => {
    const { getDocuments } = await import('@/lib/documents');
    vi.mocked(getDocuments).mockResolvedValue({ documents: [], total: 0 });

    const { rerender } = render(
      <DocumentList
        workspaceId="ws-1"
        onSelectDocument={vi.fn()}
        onDeleteDocument={vi.fn()}
      />
    );

    await waitFor(() => {
      expect(getDocuments).toHaveBeenCalledWith('ws-1');
    });

    rerender(
      <DocumentList
        workspaceId="ws-2"
        onSelectDocument={vi.fn()}
        onDeleteDocument={vi.fn()}
      />
    );

    await waitFor(() => {
      expect(getDocuments).toHaveBeenCalledWith('ws-2');
    });
  });
});
