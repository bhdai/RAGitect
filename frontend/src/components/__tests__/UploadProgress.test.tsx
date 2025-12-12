import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { UploadProgress } from '../UploadProgress';

describe('UploadProgress', () => {
  const mockOnCancel = vi.fn();

  it('renders upload progress for multiple files', () => {
    const uploads = [
      { fileName: 'file1.txt', progress: 50, status: 'uploading' as const },
      { fileName: 'file2.md', progress: 100, status: 'success' as const },
      { fileName: 'file3.pdf', progress: 0, status: 'error' as const },
    ];

    render(<UploadProgress uploads={uploads} onCancel={mockOnCancel} />);

    expect(screen.getByText('file1.txt')).toBeInTheDocument();
    expect(screen.getByText('file2.md')).toBeInTheDocument();
    expect(screen.getByText('file3.pdf')).toBeInTheDocument();
  });

  it('shows progress bars for uploading files', () => {
    const uploads = [
      { fileName: 'file1.txt', progress: 75, status: 'uploading' as const },
    ];

    const { container } = render(
      <UploadProgress uploads={uploads} onCancel={mockOnCancel} />
    );

    const progressBar = container.querySelector('[role="progressbar"]');
    expect(progressBar).toBeInTheDocument();
  });

  it('displays success state for completed uploads', () => {
    const uploads = [
      { fileName: 'success.txt', progress: 100, status: 'success' as const },
    ];

    render(<UploadProgress uploads={uploads} onCancel={mockOnCancel} />);

    expect(screen.getByText('success.txt')).toBeInTheDocument();
    // Success indicator should be visible
    const successIcon = screen.getByTestId('success-icon');
    expect(successIcon).toBeInTheDocument();
  });

  it('displays error state for failed uploads', () => {
    const uploads = [
      {
        fileName: 'error.txt',
        progress: 0,
        status: 'error' as const,
        error: 'Upload failed',
      },
    ];

    render(<UploadProgress uploads={uploads} onCancel={mockOnCancel} />);

    expect(screen.getByText('error.txt')).toBeInTheDocument();
    expect(screen.getByText('Upload failed')).toBeInTheDocument();
  });

  it('calls onCancel when cancel button clicked', () => {
    const uploads = [
      { fileName: 'file1.txt', progress: 50, status: 'uploading' as const },
    ];

    render(<UploadProgress uploads={uploads} onCancel={mockOnCancel} />);

    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    fireEvent.click(cancelButton);

    expect(mockOnCancel).toHaveBeenCalledWith('file1.txt');
  });

  it('shows file size when provided', () => {
    const uploads = [
      {
        fileName: 'file1.txt',
        progress: 50,
        status: 'uploading' as const,
        size: 1024,
      },
    ];

    render(<UploadProgress uploads={uploads} onCancel={mockOnCancel} />);

    expect(screen.getByText(/1.0 KB/i)).toBeInTheDocument();
  });

  it('hides cancel button for completed uploads', () => {
    const uploads = [
      { fileName: 'success.txt', progress: 100, status: 'success' as const },
    ];

    render(<UploadProgress uploads={uploads} onCancel={mockOnCancel} />);

    const cancelButton = screen.queryByRole('button', { name: /cancel/i });
    expect(cancelButton).not.toBeInTheDocument();
  });

  it('allows retry for failed uploads', () => {
    const mockOnRetry = vi.fn();
    const uploads = [
      {
        fileName: 'error.txt',
        progress: 0,
        status: 'error' as const,
        error: 'Network error',
      },
    ];

    render(
      <UploadProgress uploads={uploads} onCancel={mockOnCancel} onRetry={mockOnRetry} />
    );

    const retryButton = screen.getByRole('button', { name: /retry/i });
    fireEvent.click(retryButton);

    expect(mockOnRetry).toHaveBeenCalledWith('error.txt');
  });
});
