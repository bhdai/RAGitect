/**
 * DocumentList component
 * 
 * Displays a list of documents in a workspace with color-coded status,
 * selection, and delete functionality. Supports collapsed (icon-only) mode.
 */

'use client';

import { useEffect, useState, useCallback } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from '@/components/ui/tooltip';
import { getDocuments, type Document } from '@/lib/documents';
import { FileText, Trash2, Loader2, Globe, Play, FileType } from 'lucide-react';
import { cn } from '@/lib/utils';

interface DocumentListProps {
  workspaceId: string;
  onSelectDocument: (document: Document) => void;
  onDeleteDocument: (document: Document) => void;
  refreshTrigger?: number; // Increment to trigger refresh
  collapsed?: boolean; // Icon-only mode for collapsed sidebar
}

/**
 * Get styling classes based on document status
 * - ready: normal/default styling
 * - processing/embedding: subtle pulsing border
 * - error: red border/background
 */
function getStatusStyles(status: string) {
  switch (status) {
    case 'ready':
      return {
        card: '',
        icon: 'text-zinc-500',
      };
    case 'processing':
    case 'embedding':
      return {
        card: 'border-blue-300 dark:border-blue-700 animate-pulse',
        icon: 'text-blue-500',
      };
    case 'error':
      return {
        card: 'border-red-300 dark:border-red-700 bg-red-50 dark:bg-red-950/30',
        icon: 'text-red-500',
      };
    default:
      return {
        card: 'border-zinc-300 dark:border-zinc-600',
        icon: 'text-zinc-400',
      };
  }
}

/**
 * Get the appropriate icon component for a document based on its source type.
 * URL documents have null fileType, so we detect type from fileName pattern.
 */
function getDocumentIcon(doc: Document, className: string) {
  const name = doc.fileName.toLowerCase();

  // YouTube URLs
  if (name.includes('youtube.com/watch') || name.includes('youtu.be/')) {
    return <Play className={className} data-testid="doc-icon-youtube" />;
  }

  // PDF URLs (null fileType + .pdf in URL)
  if (doc.fileType === null && (name.endsWith('.pdf') || name.includes('.pdf?'))) {
    return <FileType className={className} data-testid="doc-icon-pdf" />;
  }

  // Web URLs (null fileType, not YouTube/PDF)
  if (doc.fileType === null) {
    return <Globe className={className} data-testid="doc-icon-globe" />;
  }

  // Default: regular file upload
  return <FileText className={className} data-testid="doc-icon-file" />;
}

export function DocumentList({
  workspaceId,
  onSelectDocument,
  onDeleteDocument,
  refreshTrigger,
  collapsed = false,
}: DocumentListProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDocuments = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await getDocuments(workspaceId);
      setDocuments(response.documents);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load documents';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [workspaceId]);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments, refreshTrigger]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="h-6 w-6 animate-spin text-zinc-400" />
        <span className="ml-2 text-sm text-zinc-500">Loading documents...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-center dark:border-red-800 dark:bg-red-950">
        <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        <Button variant="outline" size="sm" className="mt-2" onClick={fetchDocuments}>
          Try Again
        </Button>
      </div>
    );
  }

  if (documents.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-zinc-300 p-8 text-center dark:border-zinc-700">
        <FileText className="mx-auto h-10 w-10 text-zinc-400" />
        {!collapsed && (
          <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
            No documents yet. Upload some files to get started.
          </p>
        )}
      </div>
    );
  }

  // Collapsed mode: icon-only with tooltips
  if (collapsed) {
    return (
      <TooltipProvider>
        <div className="space-y-1">
          {documents.map((doc) => {
            const styles = getStatusStyles(doc.status);
            return (
              <Tooltip key={doc.id}>
                <TooltipTrigger asChild>
                  <button
                    className={cn(
                      "w-8 h-8 flex items-center justify-center rounded hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors",
                      doc.status === 'error' && 'bg-red-50 dark:bg-red-950/30'
                    )}
                    onClick={() => onSelectDocument(doc)}
                    data-testid={`collapsed-doc-${doc.id}`}
                  >
                    {getDocumentIcon(doc, cn("h-4 w-4", styles.icon))}
                  </button>
                </TooltipTrigger>
                <TooltipContent side="right">
                  <p>{doc.fileName}</p>
                  {doc.status !== 'ready' && (
                    <p className="text-xs opacity-70 capitalize">{doc.status}</p>
                  )}
                </TooltipContent>
              </Tooltip>
            );
          })}
        </div>
      </TooltipProvider>
    );
  }

  // Expanded mode: full document cards with color-coded status
  return (
    <div className="space-y-2">
      {documents.map((doc) => {
        const styles = getStatusStyles(doc.status);
        return (
          <Card
            key={doc.id}
            className={cn(
              "flex flex-row items-center gap-2 p-2 hover:bg-zinc-50 dark:hover:bg-zinc-900 cursor-pointer transition-colors",
              styles.card
            )}
            onClick={() => onSelectDocument(doc)}
          >
            {/* Icon - color indicates status */}
            {getDocumentIcon(doc, cn("h-4 w-4 flex-shrink-0", styles.icon))}
            
            {/* File name - takes full available space */}
            <p className="text-xs font-medium truncate flex-1 min-w-0" title={doc.fileName}>
              {doc.fileName}
            </p>
            
            {/* Delete button */}
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 flex-shrink-0 text-zinc-400 hover:text-red-500"
              aria-label={`Delete ${doc.fileName}`}
              onClick={(e) => {
                e.stopPropagation();
                onDeleteDocument(doc);
              }}
            >
              <Trash2 className="h-3 w-3" />
            </Button>
          </Card>
        );
      })}
    </div>
  );
}
