/**
 * Document sidebar component for three-panel layout
 * 
 * Collapsible sidebar with document list and "Add Source" button.
 * Uses modal for document upload per UX spec.
 * 
 * Story 3.0: Streaming Infrastructure - AC3, AC4
 */

'use client';

import { useState } from 'react';
import { Plus, ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { UploadModal } from '@/components/UploadModal';
import { DocumentList } from '@/components/DocumentList';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import { cn } from '@/lib/utils';
import type { Document } from '@/lib/documents';
import type { Upload } from '@/components/UploadProgress';

export interface DocumentSidebarProps {
  /** Workspace ID for document operations */
  workspaceId: string;
  /** Callback when a document is selected for viewing */
  onSelectDocument: (doc: Document) => void;
  /** Callback when delete is requested for a document */
  onDeleteDocument: (doc: Document) => void;
  /** Counter to trigger document list refresh */
  refreshTrigger: number;
  /** Current uploads in progress (kept for interface compatibility) */
  uploads: Upload[];
  /** Callback when files are selected for upload */
  onFilesSelected: (files: File[]) => void;
  /** Callback when upload completes successfully */
  onUploadComplete: (documents: Document[]) => void;
  /** Callback when upload fails */
  onUploadError: (error: Error) => void;
  /** Callback to cancel an upload (kept for interface compatibility) */
  onCancelUpload: (fileName: string) => void;
  /** Callback to retry a failed upload (kept for interface compatibility) */
  onRetryUpload: (fileName: string) => void;
}

/**
 * Collapsible document sidebar for the three-panel workspace layout.
 * 
 * Displays:
 * - Collapse/expand toggle at top
 * - "Add Source" button (opens upload modal)
 * - Upload progress (when uploads are active)
 * - List of documents in the workspace
 */
export function DocumentSidebar({
  workspaceId,
  onSelectDocument,
  onDeleteDocument,
  refreshTrigger,
  uploads: _uploads = [], // eslint-disable-line @typescript-eslint/no-unused-vars -- Kept for interface compatibility
  onFilesSelected,
  onUploadComplete,
  onUploadError,
  onCancelUpload: _onCancelUpload = () => { }, // eslint-disable-line @typescript-eslint/no-unused-vars -- Kept for interface compatibility
  onRetryUpload: _onRetryUpload = () => { }, // eslint-disable-line @typescript-eslint/no-unused-vars -- Kept for interface compatibility
}: DocumentSidebarProps) {
  const [isCollapsed, setIsCollapsed] = useLocalStorage('sidebar-collapsed', false);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);

  const toggleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  };

  return (
    <>
      <aside
        data-testid="document-sidebar"
        className={cn(
          'flex-shrink-0 h-full flex flex-col transition-all duration-200',
          isCollapsed ? 'w-12' : 'w-80'
        )}
      >
        <div className={cn('flex flex-col h-full overflow-hidden', isCollapsed ? 'p-2' : 'p-4')}>
          {/* Header with collapse toggle */}
          <div className={cn(
            'flex items-center mb-4',
            isCollapsed ? 'justify-center' : 'justify-between'
          )}>
            {!isCollapsed && (
              <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
                Documents
              </h2>
            )}
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleCollapse}
              className="h-8 w-8"
              data-testid="sidebar-toggle"
              aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            >
              {isCollapsed ? (
                <ChevronRight className="h-4 w-4" />
              ) : (
                <ChevronLeft className="h-4 w-4" />
              )}
            </Button>
          </div>

          {/* Add Source button */}
          <Button
            variant="outline"
            onClick={() => setIsUploadModalOpen(true)}
            className={cn(
              'mb-4 flex items-center',
              isCollapsed ? 'w-8 h-8 p-0' : 'w-full justify-center'
            )}
            data-testid="add-source-button"
          >
            <Plus className={cn('h-4 w-4', !isCollapsed && 'mr-2')} />
            {!isCollapsed && <span>Add Source</span>}
          </Button>

          {/* Document list - scrollable area */}
          <div className="flex-1 min-h-0 overflow-y-auto -mr-2 pr-2">
            <DocumentList
              workspaceId={workspaceId}
              onSelectDocument={onSelectDocument}
              onDeleteDocument={onDeleteDocument}
              refreshTrigger={refreshTrigger}
              collapsed={isCollapsed}
            />
          </div>
        </div>
      </aside>

      {/* Upload Modal */}
      <UploadModal
        open={isUploadModalOpen}
        onOpenChange={setIsUploadModalOpen}
        workspaceId={workspaceId}
        onFilesSelected={onFilesSelected}
        onUploadComplete={onUploadComplete}
        onUploadError={onUploadError}
      />
    </>
  );
}
