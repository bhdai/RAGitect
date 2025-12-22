/**
 * Workspace detail page - Three-panel layout
 *
 * Displays a single workspace with:
 * - Left sidebar (280px): Document upload and listing
 * - Center panel (flex): Chat interface
 * - Right panel (700px, conditional): Document viewer
 */

'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { DocumentSidebar } from '@/components/DocumentSidebar';
import { ChatPanel } from '@/components/ChatPanel';
import { DocumentViewer } from '@/components/DocumentViewer';
import { DeleteDocumentDialog } from '@/components/DeleteDocumentDialog';
import { ProviderSelectionProvider } from '@/contexts/ProviderSelectionContext';
import type { Upload } from '@/components/UploadProgress';
import { getWorkspace } from '@/lib/api';
import { uploadDocuments, getDocumentStatus, deleteDocument, type Document } from '@/lib/documents';
import { toast } from 'sonner';
import type { Workspace } from '@/lib/types';
import type { CitationData } from '@/types/citation';

export default function WorkspacePage() {
  // Next.js App Router: use useParams hook for client components
  const params = useParams<{ id: string }>();
  const id = params.id;
  const [workspace, setWorkspace] = useState<Workspace | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploads, setUploads] = useState<Upload[]>([]);
  // Use ref for polling intervals to avoid stale closure issues
  const pollingIntervalsRef = useRef<Map<string, NodeJS.Timeout>>(new Map());

  // Document viewing and deletion state
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);
  const [documentToDelete, setDocumentToDelete] = useState<Document | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [documentListRefresh, setDocumentListRefresh] = useState(0);


  useEffect(() => {
    async function fetchWorkspace() {
      setIsLoading(true);
      setError(null);

      try {
        const data = await getWorkspace(id);
        setWorkspace(data);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load workspace';
        setError(message);
      } finally {
        setIsLoading(false);
      }
    }

    fetchWorkspace();
  }, [id]);

  // Cleanup polling intervals on unmount
  useEffect(() => {
    const intervalsRef = pollingIntervalsRef;
    return () => {
      intervalsRef.current.forEach(interval => clearInterval(interval));
      intervalsRef.current.clear();
    };
  }, []);

  const pollDocumentStatus = useCallback(async (documentId: string, fileName: string) => {
    try {
      const status = await getDocumentStatus(documentId);

      // Update upload status based on processing state
      setUploads(prev =>
        prev.map(upload =>
          upload.fileName === fileName
            ? {
              ...upload,
              status: (status.status === 'processing' || status.status === 'embedding') ? 'uploading' :
                status.status === 'ready' ? 'success' :
                  status.status === 'error' ? 'error' : upload.status,
              progress: (status.status === 'processing' || status.status === 'embedding') ? 95 :
                status.status === 'ready' ? 100 : upload.progress,
              phase: status.phase ?? undefined, // Pass phase for detailed progress indicator
            }
            : upload
        )
      );

      // Stop polling if status is terminal
      if (status.status === 'ready' || status.status === 'error') {
        const interval = pollingIntervalsRef.current.get(documentId);
        if (interval) {
          clearInterval(interval);
          pollingIntervalsRef.current.delete(documentId);
        }

        // Refresh document list when status becomes terminal
        setDocumentListRefresh(prev => prev + 1);

        if (status.status === 'ready') {
          toast.success(`Document ready: ${fileName}`);
        } else if (status.status === 'error') {
          toast.error(`Processing failed: ${fileName}`);
        }
      }
    } catch (err) {
      console.error(`Failed to poll status for ${documentId}:`, err);
    }
  }, []);

  const handleUploadComplete = useCallback((documents: Document[]) => {
    // Update uploads to show upload complete (100%)
    setUploads(prev =>
      prev.map(upload => ({
        ...upload,
        progress: 100,
        status: 'success' as const,
      }))
    );

    toast.success(`Uploaded ${documents.length} ${documents.length === 1 ? 'file' : 'files'} - parsing...`);

    // Refresh document list after upload
    setDocumentListRefresh(prev => prev + 1);

    // Start polling for each document
    documents.forEach(doc => {
      const interval = setInterval(() => {
        pollDocumentStatus(doc.id, doc.fileName);
      }, 2000); // Poll every 2 seconds

      pollingIntervalsRef.current.set(doc.id, interval);

      // Initial poll
      pollDocumentStatus(doc.id, doc.fileName);
    });

    // Don't auto-clear uploads anymore - wait for processing to complete
  }, [pollDocumentStatus]);

  const handleUploadError = (error: Error) => {
    // Mark all uploads as error
    setUploads(prev =>
      prev.map(upload => ({
        ...upload,
        status: 'error' as const,
        error: error.message,
      }))
    );
  };

  const handleCancel = (fileName: string) => {
    // Remove the upload from the list
    setUploads(prev => prev.filter(upload => upload.fileName !== fileName));
    toast.info(`Cancelled upload of ${fileName}`);
  };

  const handleRetry = async (_fileName: string) => { // eslint-disable-line @typescript-eslint/no-unused-vars
    // For now, just show a message - full retry logic would need file reference
    toast.info('Retry functionality will be available in the next iteration');
  };

  // Handle file selection from dropzone
  const handleFilesSelected = async (files: File[]) => {
    if (!workspace) return;

    // Initialize uploads
    const newUploads: Upload[] = files.map(file => ({
      fileName: file.name,
      progress: 0,
      status: 'uploading' as const,
      size: file.size,
    }));

    setUploads(newUploads);

    try {
      // Simulate progress (since we don't have real progress tracking yet)
      const progressInterval = setInterval(() => {
        setUploads(prev =>
          prev.map(upload => ({
            ...upload,
            progress: Math.min(upload.progress + 10, 90), // Cap at 90% until complete
          }))
        );
      }, 200);

      // Perform the upload
      const response = await uploadDocuments(workspace.id, files);

      clearInterval(progressInterval);
      handleUploadComplete(response.documents);
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Upload failed');
      handleUploadError(error);
      toast.error(error.message);
    }
  };

  // Handle document selection for viewing
  const handleSelectDocument = (doc: Document) => {
    setSelectedDocumentId(doc.id);
  };

  // Handle citation click for deep-dive navigation
  // Opens the document viewer for the cited document
  const handleCitationClick = useCallback((citation: CitationData) => {
    // AC6: Validate documentId exists
    if (!citation.documentId) {
      toast.error('Citation is missing document reference');
      return;
    }

    // AC1, AC5: Open viewer with correct document
    setSelectedDocumentId(citation.documentId);
  }, []);

  // Handle document deletion request (opens dialog)
  const handleDeleteDocument = (doc: Document) => {
    setDocumentToDelete(doc);
  };

  // Handle delete confirmation
  const handleDeleteConfirm = async () => {
    if (!documentToDelete) return;

    setIsDeleting(true);
    try {
      await deleteDocument(documentToDelete.id);
      toast.success(`Deleted ${documentToDelete.fileName}`);

      // Close viewer if deleted document was being viewed
      if (selectedDocumentId === documentToDelete.id) {
        setSelectedDocumentId(null);
      }

      // Refresh document list
      setDocumentListRefresh(prev => prev + 1);
      setDocumentToDelete(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete document';
      toast.error(message);
    } finally {
      setIsDeleting(false);
    }
  };

  // Handle delete cancellation
  const handleDeleteCancel = () => {
    setDocumentToDelete(null);
  };

  // Handle document viewer close
  const handleCloseViewer = () => {
    setSelectedDocumentId(null);
  };

  if (isLoading) {
    return (
      <div className="h-screen flex items-center justify-center bg-zinc-50 dark:bg-zinc-950">
        <div className="text-center">
          <div className="mx-auto h-8 w-8 animate-spin rounded-full border-4 border-zinc-300 border-t-zinc-900 dark:border-zinc-600 dark:border-t-zinc-100" />
          <p className="mt-4 text-sm text-zinc-600 dark:text-zinc-400">
            Loading workspace...
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-screen flex items-center justify-center bg-zinc-50 dark:bg-zinc-950">
        <div className="mx-auto max-w-2xl px-4">
          <div className="rounded-lg border border-red-200 bg-red-50 p-6 text-center dark:border-red-800 dark:bg-red-950">
            <p className="text-red-600 dark:text-red-400">{error}</p>
            <Link href="/">
              <Button variant="outline" className="mt-4">
                Back to Dashboard
              </Button>
            </Link>
          </div>
        </div>
      </div>
    );
  }

  if (!workspace) {
    return null;
  }

  return (
    <div className="h-screen flex flex-col overflow-hidden" style={{ backgroundColor: 'var(--workspace-bg)' }}>
      {/* Fixed header with workspace info - blends with background */}
      <header className="flex-shrink-0" style={{ backgroundColor: 'var(--workspace-bg)' }}>
        <div className="px-[var(--workspace-padding)] pt-[var(--workspace-padding)] pb-2 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="text-sm text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
            >
              ← Back
            </Link>
            <div className="hidden sm:block h-4 w-px bg-zinc-300 dark:bg-zinc-700" />
            <div className="hidden sm:block">
              <h1 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
                {workspace.name}
              </h1>
            </div>
          </div>
          {workspace.description && (
            <p className="hidden md:block text-sm text-muted-foreground max-w-md truncate">
              {workspace.description}
            </p>
          )}
        </div>
      </header>

      {/* Main content: three-panel floating cards layout */}
      <div
        className="flex-1 flex min-h-0 px-[var(--workspace-padding)] pb-[var(--workspace-padding)]"
        style={{ gap: 'var(--panel-gap)' }}
      >
        {/* Left: Document Sidebar Card */}
        <div
          className="flex-shrink-0 overflow-hidden"
          style={{
            borderRadius: 'var(--card-radius)',
            backgroundColor: 'var(--card-sidebar-bg)'
          }}
        >
          <DocumentSidebar
            workspaceId={workspace.id}
            onSelectDocument={handleSelectDocument}
            onDeleteDocument={handleDeleteDocument}
            refreshTrigger={documentListRefresh}
            uploads={uploads}
            onFilesSelected={handleFilesSelected}
            onUploadComplete={handleUploadComplete}
            onUploadError={handleUploadError}
            onCancelUpload={handleCancel}
            onRetryUpload={handleRetry}
          />
        </div>

        {/* Center: Chat Panel Card (flex-1) */}
        <div
          className="flex-1 min-w-0 overflow-hidden"
          style={{
            borderRadius: 'var(--card-radius)',
            backgroundColor: 'var(--card-chat-bg)'
          }}
        >
          <ProviderSelectionProvider>
            <ChatPanel workspaceId={workspace.id} onCitationClick={handleCitationClick} />
          </ProviderSelectionProvider>
        </div>

        {/* Right: Document Viewer Card (700px, conditional) */}
        {selectedDocumentId && (
          <div
            className="w-[700px] flex-shrink-0 overflow-hidden"
            style={{
              borderRadius: 'var(--card-radius)',
              backgroundColor: 'var(--card-viewer-bg)'
            }}
          >
            <DocumentViewer
              documentId={selectedDocumentId}
              onClose={handleCloseViewer}
            />
          </div>
        )}
      </div>

      {/* Delete confirmation dialog */}
      <DeleteDocumentDialog
        document={documentToDelete}
        open={documentToDelete !== null}
        onConfirm={handleDeleteConfirm}
        onCancel={handleDeleteCancel}
        isDeleting={isDeleting}
      />
    </div>
  );
}
