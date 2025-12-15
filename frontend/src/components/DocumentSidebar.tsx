/**
 * Document sidebar component for three-panel layout
 * 
 * Combines upload dropzone and document list in a collapsible sidebar.
 * Fixed width of 280px per UX spec from Epic 2 retrospective.
 * 
 * Story 3.0: Streaming Infrastructure - AC3, AC4
 */

'use client';

import { IngestionDropzone } from '@/components/IngestionDropzone';
import { UploadProgress, type Upload } from '@/components/UploadProgress';
import { DocumentList } from '@/components/DocumentList';
import type { Document } from '@/lib/documents';

export interface DocumentSidebarProps {
  /** Workspace ID for document operations */
  workspaceId: string;
  /** Callback when a document is selected for viewing */
  onSelectDocument: (doc: Document) => void;
  /** Callback when delete is requested for a document */
  onDeleteDocument: (doc: Document) => void;
  /** Counter to trigger document list refresh */
  refreshTrigger: number;
  /** Current uploads in progress */
  uploads: Upload[];
  /** Callback when files are selected for upload */
  onFilesSelected: (files: File[]) => void;
  /** Callback when upload completes successfully */
  onUploadComplete: (documents: Document[]) => void;
  /** Callback when upload fails */
  onUploadError: (error: Error) => void;
  /** Callback to cancel an upload */
  onCancelUpload: (fileName: string) => void;
  /** Callback to retry a failed upload */
  onRetryUpload: (fileName: string) => void;
}

/**
 * Document sidebar for the three-panel workspace layout.
 * 
 * Displays:
 * - Document heading
 * - Upload progress (when uploads are active)
 * - File upload dropzone
 * - List of documents in the workspace
 */
export function DocumentSidebar({
  workspaceId,
  onSelectDocument,
  onDeleteDocument,
  refreshTrigger,
  uploads,
  onFilesSelected,
  onUploadComplete,
  onUploadError,
  onCancelUpload,
  onRetryUpload,
}: DocumentSidebarProps) {
  return (
    <aside
      data-testid="document-sidebar"
      className="w-72 flex-shrink-0 border-r border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 overflow-y-auto"
    >
      <div className="p-4 space-y-4">
        <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
          Documents
        </h2>

        {/* Upload progress */}
        {uploads.length > 0 && (
          <UploadProgress
            uploads={uploads}
            onCancel={onCancelUpload}
            onRetry={onRetryUpload}
          />
        )}

        {/* Dropzone */}
        <IngestionDropzone
          workspaceId={workspaceId}
          onFilesSelected={onFilesSelected}
          onUploadComplete={onUploadComplete}
          onUploadError={onUploadError}
        />

        {/* Document list */}
        <div className="mt-4">
          <DocumentList
            workspaceId={workspaceId}
            onSelectDocument={onSelectDocument}
            onDeleteDocument={onDeleteDocument}
            refreshTrigger={refreshTrigger}
          />
        </div>
      </div>
    </aside>
  );
}
