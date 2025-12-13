/**
 * DeleteDocumentDialog Unit Tests
 * 
 * Tests dialog behavior for document deletion confirmation.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DeleteDocumentDialog } from '../DeleteDocumentDialog';

// Reset mocks before each test
beforeEach(() => {
  vi.clearAllMocks();
});

describe('DeleteDocumentDialog', () => {
  const mockDocument = {
    id: '1',
    fileName: 'test-document.pdf',
  };

  it('shows confirmation message with document name', async () => {
    render(
      <DeleteDocumentDialog
        document={mockDocument}
        open={true}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />
    );

    expect(screen.getByRole('heading', { name: /delete document/i })).toBeInTheDocument();
    expect(screen.getByText(/test-document.pdf/)).toBeInTheDocument();
  });

  it('calls onConfirm when delete button is clicked', async () => {
    const user = userEvent.setup();
    const handleConfirm = vi.fn();

    render(
      <DeleteDocumentDialog
        document={mockDocument}
        open={true}
        onConfirm={handleConfirm}
        onCancel={vi.fn()}
      />
    );

    const deleteButton = screen.getByRole('button', { name: /delete/i });
    await user.click(deleteButton);

    expect(handleConfirm).toHaveBeenCalled();
  });

  it('calls onCancel when cancel button is clicked', async () => {
    const user = userEvent.setup();
    const handleCancel = vi.fn();

    render(
      <DeleteDocumentDialog
        document={mockDocument}
        open={true}
        onConfirm={vi.fn()}
        onCancel={handleCancel}
      />
    );

    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    await user.click(cancelButton);

    expect(handleCancel).toHaveBeenCalled();
  });

  it('does not render when open is false', () => {
    render(
      <DeleteDocumentDialog
        document={mockDocument}
        open={false}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />
    );

    expect(screen.queryByText(/delete document/i)).not.toBeInTheDocument();
  });

  it('does not render when document is null', () => {
    render(
      <DeleteDocumentDialog
        document={null}
        open={true}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />
    );

    expect(screen.queryByText(/delete document/i)).not.toBeInTheDocument();
  });

  it('disables buttons when isDeleting is true', () => {
    render(
      <DeleteDocumentDialog
        document={mockDocument}
        open={true}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
        isDeleting={true}
      />
    );

    const deleteButton = screen.getByRole('button', { name: /deleting/i });
    const cancelButton = screen.getByRole('button', { name: /cancel/i });

    expect(deleteButton).toBeDisabled();
    expect(cancelButton).toBeDisabled();
  });

  it('shows deleting state in button text', () => {
    render(
      <DeleteDocumentDialog
        document={mockDocument}
        open={true}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
        isDeleting={true}
      />
    );

    expect(screen.getByText(/deleting/i)).toBeInTheDocument();
  });
});
