import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { IngestionDropzone } from '../IngestionDropzone';

describe('IngestionDropzone', () => {
  const mockOnUploadComplete = vi.fn();
  const mockOnUploadError = vi.fn();
  const workspaceId = 'test-workspace-id';

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders in idle state with drop zone text', () => {
    render(
      <IngestionDropzone
        workspaceId={workspaceId}
        onUploadComplete={mockOnUploadComplete}
        onUploadError={mockOnUploadError}
      />
    );

    expect(screen.getByText(/drop files here/i)).toBeInTheDocument();
    expect(screen.getByText(/or click to select/i)).toBeInTheDocument();
  });

  it('accepts multiple file selection via input', () => {
    render(
      <IngestionDropzone
        workspaceId={workspaceId}
        onUploadComplete={mockOnUploadComplete}
        onUploadError={mockOnUploadError}
      />
    );

    const input = screen.getByTestId('file-input') as HTMLInputElement;
    expect(input.multiple).toBe(true);
  });

  it('accepts correct file types', () => {
    render(
      <IngestionDropzone
        workspaceId={workspaceId}
        onUploadComplete={mockOnUploadComplete}
        onUploadError={mockOnUploadError}
      />
    );

    const input = screen.getByTestId('file-input') as HTMLInputElement;
    const expectedTypes = '.pdf,.docx,.txt,.md,.markdown,.pptx,.xlsx,.html,.htm';
    expect(input.accept).toBe(expectedTypes);
  });

  it('changes visual state on drag over', () => {
    const { container } = render(
      <IngestionDropzone
        workspaceId={workspaceId}
        onUploadComplete={mockOnUploadComplete}
        onUploadError={mockOnUploadError}
      />
    );

    const dropzone = container.querySelector('[data-testid="dropzone"]');
    expect(dropzone).toBeInTheDocument();

    fireEvent.dragOver(dropzone!);
    expect(dropzone).toHaveClass('border-primary');

    fireEvent.dragLeave(dropzone!);
    expect(dropzone).not.toHaveClass('border-primary');
  });

  it('handles file drop', async () => {
    render(
      <IngestionDropzone
        workspaceId={workspaceId}
        onUploadComplete={mockOnUploadComplete}
        onUploadError={mockOnUploadError}
      />
    );

    const dropzone = screen.getByTestId('dropzone');
    const file = new File(['test content'], 'test.txt', { type: 'text/plain' });

    fireEvent.drop(dropzone, {
      dataTransfer: {
        files: [file],
      },
    });

    await waitFor(() => {
      expect(screen.getByText(/test.txt/i)).toBeInTheDocument();
    });
  });

  it('validates unsupported file types', async () => {
    render(
      <IngestionDropzone
        workspaceId={workspaceId}
        onUploadComplete={mockOnUploadComplete}
        onUploadError={mockOnUploadError}
      />
    );

    const dropzone = screen.getByTestId('dropzone');
    const file = new File(['test'], 'test.exe', { type: 'application/x-msdownload' });

    fireEvent.drop(dropzone, {
      dataTransfer: {
        files: [file],
      },
    });

    await waitFor(() => {
      expect(mockOnUploadError).toHaveBeenCalled();
    });
  });

  it('enforces max file size limit', async () => {
    const maxFileSize = 1024; // 1KB for testing
    render(
      <IngestionDropzone
        workspaceId={workspaceId}
        onUploadComplete={mockOnUploadComplete}
        onUploadError={mockOnUploadError}
        maxFileSize={maxFileSize}
      />
    );

    const dropzone = screen.getByTestId('dropzone');
    const largeContent = 'x'.repeat(2048); // 2KB
    const file = new File([largeContent], 'large.txt', { type: 'text/plain' });

    fireEvent.drop(dropzone, {
      dataTransfer: {
        files: [file],
      },
    });

    await waitFor(() => {
      expect(mockOnUploadError).toHaveBeenCalled();
    });
  });

  it('handles multiple files simultaneously', async () => {
    render(
      <IngestionDropzone
        workspaceId={workspaceId}
        onUploadComplete={mockOnUploadComplete}
        onUploadError={mockOnUploadError}
      />
    );

    const dropzone = screen.getByTestId('dropzone');
    const files = [
      new File(['content1'], 'file1.txt', { type: 'text/plain' }),
      new File(['content2'], 'file2.md', { type: 'text/markdown' }),
    ];

    fireEvent.drop(dropzone, {
      dataTransfer: {
        files,
      },
    });

    await waitFor(() => {
      expect(screen.getByText(/file1.txt/i)).toBeInTheDocument();
      expect(screen.getByText(/file2.md/i)).toBeInTheDocument();
    });
  });
});
