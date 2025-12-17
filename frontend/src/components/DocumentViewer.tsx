/**
 * DocumentViewer component
 * 
 * Side panel component that displays document content in markdown format.
 * Implements FR9 "side-by-side with chat" requirement.
 */

'use client';

import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Button } from '@/components/ui/button';
import { getDocument, type DocumentDetail } from '@/lib/documents';
import { X, FileText, Loader2 } from 'lucide-react';

interface DocumentViewerProps {
  documentId: string | null;
  onClose: () => void;
}

export function DocumentViewer({ documentId, onClose }: DocumentViewerProps) {
  const [document, setDocument] = useState<DocumentDetail | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!documentId) {
      setDocument(null);
      return;
    }

    async function fetchDocument() {
      setIsLoading(true);
      setError(null);
      try {
        const doc = await getDocument(documentId!);
        setDocument(doc);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load document';
        setError(message);
      } finally {
        setIsLoading(false);
      }
    }

    fetchDocument();
  }, [documentId]);

  // Don't render anything if no document selected
  if (!documentId) {
    return null;
  }

  return (
    <div className="flex h-full flex-col border-l border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-200 px-4 py-3 dark:border-zinc-800">
        <div className="flex items-center gap-2 min-w-0">
          <FileText className="h-5 w-5 flex-shrink-0 text-zinc-500" />
          <h2 className="text-sm font-medium truncate">
            {document?.fileName ?? 'Loading...'}
          </h2>
        </div>
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          onClick={onClose}
          aria-label="Close document viewer"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {isLoading && (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-6 w-6 animate-spin text-zinc-400" />
            <span className="ml-2 text-sm text-zinc-500">Loading document...</span>
          </div>
        )}

        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-center dark:border-red-800 dark:bg-red-950">
            <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
          </div>
        )}

        {!isLoading && !error && document && (
          <>
            {document.processedContent ? (
              <article className="text-sm leading-relaxed">
                <ReactMarkdown
                  components={{
                    // Enhanced headings - bigger and bolder
                    h1: (props) => <h1 className="text-2xl font-bold mb-4 mt-6 first:mt-0" {...props} />,
                    h2: (props) => <h2 className="text-xl font-bold mb-3 mt-6" {...props} />,
                    h3: (props) => <h3 className="text-lg font-semibold mb-2 mt-4" {...props} />,
                    h4: (props) => <h4 className="text-base font-semibold mb-2 mt-3" {...props} />,

                    // Basic lists
                    ul: (props) => <ul className="list-disc pl-6 my-3 space-y-1" {...props} />,
                    ol: (props) => <ol className="list-decimal pl-6 my-3 space-y-1" {...props} />,
                    li: (props) => <li className="leading-relaxed" {...props} />,

                    // Everything else renders as default/raw
                    p: (props) => <p className="mb-3" {...props} />,
                  }}
                >
                  {document.processedContent}
                </ReactMarkdown>
              </article>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <FileText className="h-10 w-10 text-zinc-400" />
                <p className="mt-2 text-sm text-zinc-500">
                  No content available. Document may still be processing.
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
