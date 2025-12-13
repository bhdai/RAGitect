/**
 * DocumentList component
 * 
 * Displays a list of documents in a workspace with status badges,
 * selection, and delete functionality.
 */

'use client';

import { useEffect, useState, useCallback } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { getDocuments, type Document } from '@/lib/documents';
import { FileText, Trash2, Loader2 } from 'lucide-react';

interface DocumentListProps {
  workspaceId: string;
  onSelectDocument: (document: Document) => void;
  onDeleteDocument: (document: Document) => void;
  refreshTrigger?: number; // Increment to trigger refresh
}

export function DocumentList({
  workspaceId,
  onSelectDocument,
  onDeleteDocument,
  refreshTrigger,
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

  const getStatusVariant = (status: string): 'default' | 'secondary' | 'destructive' | 'outline' => {
    switch (status) {
      case 'ready':
        return 'default';
      case 'processing':
      case 'embedding':
        return 'secondary';
      case 'error':
        return 'destructive';
      default:
        return 'outline';
    }
  };

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
        <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
          No documents yet. Upload some files to get started.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {documents.map((doc) => (
        <Card
          key={doc.id}
          className="flex flex-row items-center gap-3 p-3 hover:bg-zinc-50 dark:hover:bg-zinc-900 cursor-pointer transition-colors"
          onClick={() => onSelectDocument(doc)}
        >
          {/* Icon */}
          <FileText className="h-5 w-5 flex-shrink-0 text-zinc-500" />
          
          {/* File name - takes available space but truncates */}
          <p className="text-sm font-medium truncate flex-1 min-w-0">{doc.fileName}</p>
          
          {/* Status badge */}
          <Badge variant={getStatusVariant(doc.status)} className="flex-shrink-0">
            {doc.status}
          </Badge>
          
          {/* Delete button */}
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 flex-shrink-0 text-zinc-400 hover:text-red-500"
            aria-label={`Delete ${doc.fileName}`}
            onClick={(e) => {
              e.stopPropagation();
              onDeleteDocument(doc);
            }}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </Card>
      ))}
    </div>
  );
}
