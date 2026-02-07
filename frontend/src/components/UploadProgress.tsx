/**
 * UploadProgress component
 * 
 * Displays upload progress for multiple files with individual progress bars.
 * Shows file status (uploading/success/error) with appropriate visual feedback.
 */

'use client';

import { CheckCircle2, XCircle, X, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

export type UploadStatus = 'uploading' | 'success' | 'error';

/**
 * Processing phase for detailed progress indication
 */
export type ProcessingPhase = 'fetching' | 'parsing' | 'embedding' | null;

export interface Upload {
  fileName: string;
  progress: number; // 0-100
  status: UploadStatus;
  size?: number; // bytes
  error?: string;
  phase?: ProcessingPhase; // Current processing phase
}

interface UploadProgressProps {
  uploads: Upload[];
  onCancel: (fileName: string) => void;
  onRetry?: (fileName: string) => void;
}

/**
 * Format bytes to human-readable size
 */
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

/**
 * Simple inline progress bar component
 */
function ProgressBar({ value }: { value: number }) {
  return (
    <div
      className="w-full h-2 bg-muted rounded-full overflow-hidden"
      role="progressbar"
      aria-valuenow={value}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div
        className="h-full bg-primary transition-all duration-300"
        style={{ width: `${value}%` }}
      />
    </div>
  );
}

export function UploadProgress({ uploads, onCancel, onRetry }: UploadProgressProps) {
  if (uploads.length === 0) return null;

  return (
    <div className="w-full space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">
          Uploading {uploads.length} {uploads.length === 1 ? 'file' : 'files'}
        </h3>
      </div>

      <div className="space-y-2">
        {uploads.map((upload) => (
          <div
            key={upload.fileName}
            className={cn(
              'flex flex-col gap-2 p-3 rounded-lg border',
              upload.status === 'success' && 'bg-green-50 border-green-200 dark:bg-green-950/20 dark:border-green-900',
              upload.status === 'error' && 'bg-red-50 border-red-200 dark:bg-red-950/20 dark:border-red-900',
              upload.status === 'uploading' && 'bg-muted/50'
            )}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex items-center gap-2 flex-1 min-w-0">
                {upload.status === 'success' && (
                  <CheckCircle2
                    className="h-4 w-4 text-green-600 dark:text-green-400 flex-shrink-0"
                    data-testid="success-icon"
                  />
                )}
                {upload.status === 'error' && (
                  <XCircle className="h-4 w-4 text-red-600 dark:text-red-400 flex-shrink-0" />
                )}

                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{upload.fileName}</p>
                  {upload.size && (
                    <p className="text-xs text-muted-foreground">
                      {formatBytes(upload.size)}
                    </p>
                  )}
                  {upload.status === 'uploading' && upload.progress >= 95 && (
                    <p className="text-xs text-muted-foreground mt-1" data-testid="processing-message">
                      {upload.phase === 'fetching'
                        ? 'Fetching content...'
                        : upload.phase === 'embedding' 
                          ? 'Generating Embeddings...'
                          : 'Parsing document...'}
                    </p>
                  )}
                  {upload.error && (
                    <p className="text-xs text-red-600 dark:text-red-400 mt-1">
                      {upload.error}
                    </p>
                  )}
                </div>
              </div>

              <div className="flex items-center gap-1">
                {upload.status === 'uploading' && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={() => onCancel(upload.fileName)}
                    aria-label="Cancel upload"
                  >
                    <X className="h-3 w-3" />
                  </Button>
                )}

                {upload.status === 'error' && onRetry && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={() => onRetry(upload.fileName)}
                    aria-label="Retry upload"
                  >
                    <RotateCcw className="h-3 w-3" />
                  </Button>
                )}
              </div>
            </div>

            {upload.status === 'uploading' && (
              <div className="space-y-1">
                <ProgressBar value={upload.progress} />
                <p className="text-xs text-muted-foreground text-right">
                  {upload.progress}%
                </p>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
